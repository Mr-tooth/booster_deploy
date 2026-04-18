"""T1 getup deployment policy and controller configuration."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    RobotCfg,
    PolicyCfg,
    PrepareStateCfg,
)
from booster_deploy.robots.booster import T1_23DOF_CFG
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass


def _build_t1_getup_robot_cfg() -> RobotCfg:
    """Return a T1 robot config aligned with mjlab getup actuator settings."""
    # Per-joint gains derived from mjlab_playground t1 constants.
    neck_kp = 1.7765287881361242
    neck_kd = 0.2261946708
    arm_kp = 27.884395858584607
    arm_kd = 3.5503515528768004
    waist_hip_roll_yaw_kp = 47.1890459348658
    waist_hip_roll_yaw_kd = 6.008295943125001
    hip_pitch_kp = 51.70764690749004
    hip_pitch_kd = 6.5836220883048
    knee_kp = 62.77186820000181
    knee_kd = 7.9923624980472
    ankle_kp = 33.512439059399846
    ankle_kd = 4.2669362699711995

    joint_stiffness = [
        neck_kp,
        neck_kp,
        arm_kp,
        arm_kp,
        arm_kp,
        arm_kp,
        arm_kp,
        arm_kp,
        arm_kp,
        arm_kp,
        waist_hip_roll_yaw_kp,
        hip_pitch_kp,
        waist_hip_roll_yaw_kp,
        waist_hip_roll_yaw_kp,
        knee_kp,
        ankle_kp,
        ankle_kp,
        hip_pitch_kp,
        waist_hip_roll_yaw_kp,
        waist_hip_roll_yaw_kp,
        knee_kp,
        ankle_kp,
        ankle_kp,
    ]

    joint_damping = [
        neck_kd,
        neck_kd,
        arm_kd,
        arm_kd,
        arm_kd,
        arm_kd,
        arm_kd,
        arm_kd,
        arm_kd,
        arm_kd,
        waist_hip_roll_yaw_kd,
        hip_pitch_kd,
        waist_hip_roll_yaw_kd,
        waist_hip_roll_yaw_kd,
        knee_kd,
        ankle_kd,
        ankle_kd,
        hip_pitch_kd,
        waist_hip_roll_yaw_kd,
        waist_hip_roll_yaw_kd,
        knee_kd,
        ankle_kd,
        ankle_kd,
    ]

    effort_limit = [
        7.0,
        7.0,
        36.0,
        36.0,
        36.0,
        36.0,
        36.0,
        36.0,
        36.0,
        36.0,
        40.0,
        55.0,
        40.0,
        40.0,
        65.0,
        50.0,
        50.0,
        55.0,
        40.0,
        40.0,
        65.0,
        50.0,
        50.0,
    ]

    home_joint_pos = [
        0.0,   # AAHead_yaw
        0.0,   # Head_pitch
        0.0,   # Left_Shoulder_Pitch
        -1.4,  # Left_Shoulder_Roll
        0.0,   # Left_Elbow_Pitch
        -0.4,  # Left_Elbow_Yaw
        0.0,   # Right_Shoulder_Pitch
        1.4,   # Right_Shoulder_Roll
        0.0,   # Right_Elbow_Pitch
        0.4,   # Right_Elbow_Yaw
        0.0,   # Waist
        -0.2,  # Left_Hip_Pitch
        0.0,   # Left_Hip_Roll
        0.0,   # Left_Hip_Yaw
        0.4,   # Left_Knee_Pitch
        -0.2,  # Left_Ankle_Pitch
        0.0,   # Left_Ankle_Roll
        -0.2,  # Right_Hip_Pitch
        0.0,   # Right_Hip_Roll
        0.0,   # Right_Hip_Yaw
        0.4,   # Right_Knee_Pitch
        -0.2,  # Right_Ankle_Pitch
        0.0,   # Right_Ankle_Roll
    ]

    return T1_23DOF_CFG.replace(  # type: ignore
        mjcf_path="{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.xml",
        joint_stiffness=joint_stiffness,
        joint_damping=joint_damping,
        effort_limit=effort_limit,
        prepare_state=PrepareStateCfg(
            stiffness=joint_stiffness,
            damping=joint_damping,
            joint_pos=home_joint_pos,
        ),
    )


class T1GetupPolicy(Policy):
    """Getup actor policy aligned with mjlab T1 getup observation/action semantics."""

    def __init__(self, cfg: T1GetupPolicyCfg, controller: BaseController) -> None:
        """Initialize getup actor runtime and policy state buffers.

        Args:
            cfg: T1 getup policy configuration.
            controller: Controller runtime providing robot state and timing.

        """
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot
        self.initialize_model_runtime(self.cfg.checkpoint_path)

        self.last_action = torch.zeros(
            len(self.cfg.policy_joint_names),
            dtype=torch.float32,
            device=self.robot.data.device,
        )
        self.real2sim_joint_map = torch.tensor(
            [self.robot.cfg.joint_names.index(name) for name in self.cfg.policy_joint_names],
            dtype=torch.long,
            device=self.robot.data.device,
        )

    def reset(self) -> None:
        """Reset policy state at controller startup."""
        self.last_action.zero_()

    def compute_observation(self) -> torch.Tensor:
        """Build actor observation matching mjlab getup actor term order.

        Returns:
            Actor observation tensor with shape `(75,)`.

        """
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.robot.data.device)
        projected_gravity = lab_math.quat_apply_inverse(self.robot.data.root_quat_w, gravity_w)

        mapped_default_pos = self.robot.default_joint_pos[self.real2sim_joint_map]
        mapped_dof_pos = self.robot.data.joint_pos[self.real2sim_joint_map]
        mapped_dof_vel = self.robot.data.joint_vel[self.real2sim_joint_map]

        obs = torch.cat(
            [
                self.robot.data.root_ang_vel_b * self.cfg.obs_ang_vel_scale,
                projected_gravity,
                (mapped_dof_pos - mapped_default_pos) * self.cfg.obs_dof_pos_scale,
                mapped_dof_vel * self.cfg.obs_dof_vel_scale,
                self.last_action * self.cfg.obs_last_action_scale,
            ],
            dim=0,
        )
        return torch.clamp(obs, -self.cfg.clip_obs, self.cfg.clip_obs)

    def inference(self) -> torch.Tensor:
        """Run one actor forward pass and produce relative joint-position targets.

        Returns:
            Joint target tensor in real-joint order.

        """
        obs = self.compute_observation()
        obs_batch = obs.unsqueeze(0)

        with torch.no_grad():
            action_batch = self.infer_model(obs_batch)
            if action_batch.ndim == 2 and action_batch.shape[0] == 1:
                action = action_batch.squeeze(0)
            else:
                action = action_batch.reshape(-1)
            action = torch.clamp(action, -self.cfg.clip_action, self.cfg.clip_action)

        self.last_action = action.clone()
        current_joint_pos = self.robot.data.joint_pos.clone()

        if int(getattr(self.controller, "_step_count", 0)) <= self.cfg.settle_steps:
            return current_joint_pos

        dof_targets = current_joint_pos.clone()
        dof_targets.scatter_reduce_(
            0,
            self.real2sim_joint_map,
            action * self.cfg.action_scale,
            reduce="sum",
        )
        return dof_targets


@configclass
class T1GetupPolicyCfg(PolicyCfg):
    """Configuration schema for T1 getup actor policy deployment."""

    constructor = T1GetupPolicy
    checkpoint_path: str = MISSING  # type: ignore
    policy_joint_names: list[str] = MISSING  # type: ignore
    action_scale: float = 0.6
    settle_steps: int = 50
    clip_obs: float = 100.0
    clip_action: float = 100.0
    obs_ang_vel_scale: float = 1.0
    obs_dof_pos_scale: float = 1.0
    obs_dof_vel_scale: float = 1.0
    obs_last_action_scale: float = 1.0


@configclass
class T1GetupControllerCfg(ControllerCfg):
    """Controller configuration for T1 getup task deployment."""

    robot = _build_t1_getup_robot_cfg()
    policy: T1GetupPolicyCfg = T1GetupPolicyCfg(
        policy_joint_names=list(T1_23DOF_CFG.joint_names),
    )
