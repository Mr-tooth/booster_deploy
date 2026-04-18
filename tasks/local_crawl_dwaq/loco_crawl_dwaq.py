from __future__ import annotations

from dataclasses import MISSING
import math
from typing import Any, cast

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    PolicyCfg,
    VelocityCommandCfg,
)
from booster_deploy.robots.booster import T1_23DOF_CRAWL_CFG
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass


class LocoCrawlDwaqPolicy(Policy):
    """Crawl policy aligned with the training observation pipeline."""

    def __init__(self, cfg: LocoCrawlDwaqPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        self.initialize_model_runtime(self.cfg.checkpoint_path)

        self.actor_obs_history_length = cfg.actor_obs_history_length
        self.action_scale = cfg.action_scale

        self.obs_history: torch.Tensor | None = None
        self.last_action = torch.zeros(
            len(self.cfg.policy_joint_names),
            dtype=torch.float32,
        )

        self.real2sim_joint_map = torch.tensor(
            [
                self.robot.cfg.joint_names.index(name)
                for name in self.cfg.policy_joint_names
            ],
            dtype=torch.long,
        )

        a2base_euler = torch.tensor(
            [0.0, torch.pi / 2.0, 0.0],
            dtype=torch.float32,
        )
        self.a2base_quat = lab_math.quat_from_euler_xyz(
            a2base_euler[0],
            a2base_euler[1],
            a2base_euler[2],
        )
        self.a2base_matrix = lab_math.matrix_from_euler(a2base_euler, "XYZ")

        self.command_scale = torch.tensor(
            [self.cfg.lin_vel_scale, self.cfg.lin_vel_scale,
             self.cfg.ang_vel_scale],
            dtype=torch.float32,
        )
        # DWAQ actor is two-input (obs + stacked history). We cache the
        # flattened history each step so both torch and onnx paths can reuse it.
        self._cached_obs_history: torch.Tensor | None = None
        self._onnx_obs_input_name: str | None = None
        self._onnx_hist_input_name: str | None = None
        self._resolve_onnx_input_names()

    def _resolve_onnx_input_names(self) -> None:
        """Resolve ONNX two-input names for obs and obs_history.

        Uses name heuristics first, then shape heuristics as fallback.
        """
        if self._backend != "onnx":
            return
        if len(self._onnx_input_names) < 2:
            raise RuntimeError(
                "DWAQ ONNX model must expose two inputs: obs and obs_history"
            )

        obs_name = None
        hist_name = None
        # 1) Prefer semantic name matching for robust cross-export behavior.
        for name in self._onnx_input_names:
            lower = name.lower()
            if "history" in lower:
                hist_name = name
            elif "obs" in lower:
                obs_name = name

        if obs_name is None or hist_name is None:
            # fallback for unnamed inputs, based on training dimensions
            # obs: 73, obs_history: 73 * 20 = 1460
            # 2) Fallback to expected static shapes from train cfg.
            try:
                sess_inputs = self._onnx_session.get_inputs()  # type: ignore
                for item in sess_inputs:
                    if item.shape == [1, 73]:
                        obs_name = item.name
                    elif item.shape == [1, 1460]:
                        hist_name = item.name
            except Exception:
                pass

        if obs_name is None or hist_name is None:
            # Final fallback: preserve model-declared input order.
            # This keeps old exports usable even when names are generic.
            obs_name = self._onnx_input_names[0]
            hist_name = self._onnx_input_names[1]

        self._onnx_obs_input_name = obs_name
        self._onnx_hist_input_name = hist_name

    def reset(self) -> None:
        pass

    def _compute_phase(self, time_s: float) -> float:
        return (time_s % self.cfg.cycle_time) / self.cfg.cycle_time

    def _compute_one_hot_gait(
        self,
        lin_x: float,
        lin_y: float,
        ang_z: float,
    ) -> tuple[torch.Tensor, bool]:
        lin_norm = math.sqrt(lin_x * lin_x + lin_y * lin_y)
        ang_norm = abs(ang_z)
        # Keep stand/crawl classifier consistent with training cfg thresholds.
        if (
            lin_norm < self.cfg.stand_lin_vel_threshold
            and ang_norm < self.cfg.stand_ang_vel_threshold
        ):
            return torch.tensor([1.0, -1.0], dtype=torch.float32), True  # Standing gait
        return torch.tensor([-1.0, 1.0], dtype=torch.float32), False  # Crawling gait

    def compute_observation(self) -> tuple[torch.Tensor, torch.Tensor]:
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        world_base_quat = self.robot.data.root_quat_w
        base_ang_vel_b = self.robot.data.root_ang_vel_b

        # Keep observation aligned with training pipeline in
        # T1_crawl_dwaq.compute_observations() (single obs dim = 73).
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        projected_gravity_crawl = lab_math.quat_apply_inverse(
            world_base_quat,
            gravity_w,
        )
        base_ang_vel_crawl = base_ang_vel_b

        vel_cmd = self.controller.vel_command
        if vel_cmd is None:
            lin_vel_x = 0.0
            lin_vel_y = 0.0
            ang_vel_yaw = 0.0
        else:
            lin_vel_x = vel_cmd.lin_vel_x
            lin_vel_y = vel_cmd.lin_vel_y
            ang_vel_yaw = vel_cmd.ang_vel_yaw

        commands = torch.tensor(
            [lin_vel_x, lin_vel_y, ang_vel_yaw],
            dtype=torch.float32,
        )
        one_hot_gait, is_standing = self._compute_one_hot_gait(
            lin_vel_x,
            lin_vel_y,
            ang_vel_yaw,
        )
        if is_standing:
            # Match training: standing samples are forced to zero command.
            commands.zero_()

        phase = self._compute_phase(self.controller.get_time())
        phase = phase % 1.0 # Ensure phase is in [0, 1)
        if(is_standing):
            # Match training behavior: standing phase is pinned.
            phase = 0.125
        
        signal = torch.tensor(
            [
                math.sin(2.0 * math.pi * phase),
                math.cos(2.0 * math.pi * phase),
            ],
            dtype=torch.float32,
        )
        
        

        mapped_default_pos = self.robot.default_joint_pos[self.real2sim_joint_map]
        mapped_dof_pos = dof_pos[self.real2sim_joint_map]
        mapped_dof_vel = dof_vel[self.real2sim_joint_map]

        obs = torch.cat(
            [
                base_ang_vel_crawl * self.cfg.ang_vel_scale,
                projected_gravity_crawl,
                commands * self.command_scale,
                signal,
                one_hot_gait,
                (mapped_dof_pos - mapped_default_pos) * self.cfg.dof_pos_scale,
                mapped_dof_vel * self.cfg.dof_vel_scale,
                self.last_action,
            ],
            dim=0,
        )
        

        if self.obs_history is None:
            self.obs_history = torch.zeros(
                (self.actor_obs_history_length, obs.shape[0]),
                dtype=torch.float32,
            )
        # History stack shape: (20, 73) -> flatten to (1460,).
        self.obs_history = self.obs_history.roll(-1, dims=0)
        self.obs_history[-1] = obs.clamp(-self.cfg.clip_obs, self.cfg.clip_obs)
        
        self.log_named_vector("ang_vel_crawl", base_ang_vel_crawl)
        self.log_named_vector("projected_gravity_crawl", projected_gravity_crawl)
        self.log_named_vector("commands", commands)
        self.log_named_vector("one_hot_gait", one_hot_gait)
        self.log_named_vector("signal", signal)
        self.log_named_vector("mapped_dof_pos", mapped_dof_pos)
        self.log_named_vector("mapped_dof_vel", mapped_dof_vel)
        self.log_named_vector("last_action", self.last_action)

        return obs, self.obs_history.flatten()

    def inference(self) -> torch.Tensor:
        obs, obs_history = self.compute_observation()
        # Cache history before model call. Both torch and onnx branches rely on
        # the same value so behavior stays identical across backends.
        self._cached_obs_history = obs_history
        with torch.no_grad():
            # infer_model() consumes cached history to keep torch and onnx
            # execution paths identical.
            action = self.infer_model(obs).squeeze(0)
            action = torch.clamp(
                action,
                -self.cfg.clip_action,
                self.cfg.clip_action,
            )

        self.last_action = action.clone()

        dof_targets = self.robot.default_joint_pos.clone()
        dof_targets.scatter_reduce_(
            0,
            self.real2sim_joint_map,
            action * self.action_scale,
            reduce="sum",
        )
        return dof_targets
    
    def infer_model(self, obs: torch.Tensor) -> torch.Tensor:
        """Run model using current obs and cached history obs.

        DWAQ actor expects two inputs: ``obs`` and ``obs_history``.
        """
        obs_history = self._cached_obs_history
        if obs_history is None:
            raise RuntimeError("obs_history cache is empty before inference")

        if self._backend == "onnx":
            if self._onnx_session is None:
                raise RuntimeError("ONNX session is not initialized")
            # ONNX two-input feed is assembled in prepare_onnx_inputs().
            outputs = self._onnx_session.run(
                self._onnx_output_names,
                self.prepare_onnx_inputs(obs),
            )
            action = self.parse_onnx_outputs(outputs)
        else:
            model = cast(torch.jit.ScriptModule, self._model)
            # TorchScript actor signature: model(obs, obs_history).
            action = model(obs, obs_history)

        if self.cfg.stat_log_inference_io:
            self.log_stats(
                {
                    "inference_input_obs": obs,
                    "inference_input_obs_history": obs_history,
                    "inference_output": action,
                }
            )
        return action

    def prepare_onnx_inputs(self, obs: torch.Tensor) -> dict[str, Any]:
        """Build ONNX input feed dict for two-input models.

        Override for multi-input models to map tensors to all ONNX input names.
        """
        if not self._onnx_input_names:
            raise RuntimeError("ONNX inputs are not initialized")
        if len(self._onnx_input_names) < 2:
            raise RuntimeError(
                "DWAQ ONNX model must expose two inputs: obs and obs_history"
            )
        obs_history = self._cached_obs_history
        if obs_history is None:
            raise RuntimeError("obs_history cache is empty for ONNX input")
        obs_name = self._onnx_obs_input_name
        hist_name = self._onnx_hist_input_name
        if obs_name is None or hist_name is None:
            raise RuntimeError("ONNX input names are unresolved")

        # ORT requires exact rank/shape. Exported model expects:
        # - obs: [1, 73]
        # - obs_history: [1, 1460]
        # For a flattened tensor from runtime, ndim is 1, so we add a batch
        # dimension via reshape(1, -1). Without this, ORT raises rank mismatch.
        obs_np = obs.detach().cpu().numpy().astype("float32")
        obs_hist_np = obs_history.detach().cpu().numpy().astype("float32")
        if obs_np.ndim == 1: #(73, ) -> (1, 73)
            obs_np = obs_np.reshape(1, -1)
        if obs_hist_np.ndim == 1: #(1460, ) -> (1, 1460)
            obs_hist_np = obs_hist_np.reshape(1, -1)
        # Final feed keys use resolved names to tolerate export-side renaming.
        return {
            obs_name: obs_np,
            hist_name: obs_hist_np,
        }


@configclass
class LocoCrawlDwaqPolicyCfg(PolicyCfg):
    constructor = LocoCrawlDwaqPolicy
    checkpoint_path: str = MISSING  # type: ignore
    policy_joint_names: list[str] = MISSING  # type: ignore

    actor_obs_history_length: int = 20
    action_scale: float = 0.25

    cycle_time: float = 0.5

    lin_vel_scale: float = 1.0
    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    clip_obs: float = 100.0
    clip_action: float = 100.0

    stand_lin_vel_threshold: float = 0.12
    stand_ang_vel_threshold: float = 0.08


@configclass
class T1CrawlDwaqControllerCfg(ControllerCfg):
    policy_dt = 0.01
    robot = T1_23DOF_CRAWL_CFG.replace(  # type: ignore
        default_joint_pos=[
            0.0, 0.0,
            -1.9198622, -1.2217305, 0.0, -1.7453293,
            -1.9198622, 1.2217305, 0.0, 1.7453293,
            0.0,
            -0.87266463, 0.0, 0.0, 1.7453293, 0.0, 0.0,
            -0.87266463, 0.0, 0.0, 1.7453293, 0.0, 0.0,
        ],
        joint_stiffness=[
            4.0, 4.0,
            150.0, 150.0, 150.0, 150.0,
            150.0, 150.0, 150.0, 150.0,
            200.0,
            # 150.0, 150.0, 150.0, 100.0, 35.0, 35.0,
            # 150.0, 150.0, 150.0, 100.0, 35.0, 35.0,
            200.0, 200.0, 200.0, 100.0, 35.0, 35.0,
            200.0, 200.0, 200.0, 100.0, 35.0, 35.0,
        ],
        joint_damping=[
            1.0, 1.0,
            # 1.0, 1.0, 1.0, 1.0,
            # 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0,
            5.0,
            # 1.5, 1.5, 1.5, 1.0, 1.0, 1.0,
            # 1.5, 1.5, 1.5, 1.0, 1.0, 1.0,
            2.5, 2.5, 2.5, 1.5, 1.5, 1.5,
            2.5, 2.5, 2.5, 1.5, 1.5, 1.5,
        ],
    )
    vel_command: VelocityCommandCfg | None = VelocityCommandCfg(
        vx_max=0.6,
        vy_max=0.6,
        vyaw_max=1.0,
    )
    policy: PolicyCfg = LocoCrawlDwaqPolicyCfg(
        policy_joint_names=[
            "Left_Shoulder_Pitch",
            "Left_Shoulder_Roll",
            "Left_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "Right_Shoulder_Pitch",
            "Right_Shoulder_Roll",
            "Right_Elbow_Pitch",
            "Right_Elbow_Yaw",
            "Left_Hip_Pitch",
            "Left_Hip_Roll",
            "Left_Hip_Yaw",
            "Left_Knee_Pitch",
            "Left_Ankle_Pitch",
            "Left_Ankle_Roll",
            "Right_Hip_Pitch",
            "Right_Hip_Roll",
            "Right_Hip_Yaw",
            "Right_Knee_Pitch",
            "Right_Ankle_Pitch",
            "Right_Ankle_Roll",
        ],
    )
    
    def __post_init__(self):
        self.mujoco.decimation = 10
        super().__post_init__()
        self.policy.stat_log_path = "logs/loco_crawl_dwaq/inference_stats.csv"
        self.policy.stat_log_inference_io = True
