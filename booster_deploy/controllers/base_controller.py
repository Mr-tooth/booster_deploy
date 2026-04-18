from __future__ import annotations
from abc import abstractmethod
import csv
import inspect
import os
from pathlib import Path
from typing import Any
from typing import cast
import torch
import numpy as np

from .controller_cfg import (
    ControllerCfg, PolicyCfg, RobotCfg, VelocityCommandCfg
)


class RobotData:
    """
    The joint indexing follows the real robot,
    described in RobotCfg.joint_names
    """

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    feedback_torque: torch.Tensor
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_b: torch.Tensor
    root_ang_vel_b: torch.Tensor

    def __init__(self, cfg: RobotCfg) -> None:
        """Allocate and initialize runtime robot state buffers.

        Args:
            cfg: Static robot configuration that defines joint/body ordering.

        Notes:
            - Internal tensors are initialized on CPU and can be moved with
              ``to(...)``.
            - ``real2sim_joint_indexes`` and ``sim2real_joint_indexes`` are
              computed once and reused to convert ordering between deployment
              interfaces and simulation-trained policies.
        """
        self.cfg = cfg
        num_joints = len(self.cfg.joint_names)
        self.real2sim_joint_indexes = [cfg.joint_names.index(name) for name in cfg.sim_joint_names]
        self.sim2real_joint_indexes = [cfg.sim_joint_names.index(name) for name in cfg.joint_names]
        self.device = "cpu"

        self.joint_pos: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.joint_vel: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.feedback_torque: torch.Tensor = torch.zeros(num_joints, dtype=torch.float32)
        self.root_lin_vel_b: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_ang_vel_b: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_pos_w: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.root_quat_w: torch.Tensor = torch.zeros(4, dtype=torch.float32)

    def to(self, device: torch.device | str) -> None:
        """Move all robot state tensors to the given device in-place.

        Args:
            device: Target torch device (e.g. ``"cpu"`` or ``"cuda:0"``).
        """
        self.device = device
        self.joint_pos = self.joint_pos.to(device)
        self.joint_vel = self.joint_vel.to(device)
        self.feedback_torque = self.feedback_torque.to(device)
        self.root_lin_vel_b = self.root_lin_vel_b.to(device)
        self.root_ang_vel_b = self.root_ang_vel_b.to(device)
        self.root_pos_w = self.root_pos_w.to(device)
        self.root_quat_w = self.root_quat_w.to(device)


class BoosterRobot:
    cfg: RobotCfg
    data: RobotData
    joint_stiffness: torch.Tensor
    joint_damping: torch.Tensor
    default_joint_pos: torch.Tensor

    def __init__(self, cfg: RobotCfg) -> None:
        self.cfg = cfg
        self.data = RobotData(cfg)

        self.joint_stiffness = torch.tensor(cfg.joint_stiffness, dtype=torch.float32)

        self.joint_damping = torch.tensor(cfg.joint_damping, dtype=torch.float32)

        self.default_joint_pos = torch.tensor(cfg.default_joint_pos, dtype=torch.float32)
        self.effort_limit = torch.tensor(cfg.effort_limit, dtype=torch.float32)

    @property
    def num_joints(self) -> int:
        return len(self.cfg.joint_names)

    @property
    def num_bodies(self) -> int:
        return len(self.cfg.body_names)


class Commands:
    pass


class VelocityCommand(Commands):
    lin_vel_x: float
    lin_vel_y: float
    ang_vel_yaw: float

    def __init__(self, cfg: VelocityCommandCfg) -> None:
        """Initialize normalized velocity command container.

        Args:
            cfg: Velocity command bounds used to scale user command input.
        """
        self.vx_max = cfg.vx_max
        self.vy_max = cfg.vy_max
        self.vyaw_max = cfg.vyaw_max

        self.lin_vel_x: float = 0.0
        self.lin_vel_y: float = 0.0
        self.ang_vel_yaw: float = 0.0


class Policy:
    def __init__(self, cfg: PolicyCfg, controller: BaseController):
        """Create a policy bound to a controller runtime.

        Args:
            cfg: Policy configuration including checkpoint path and device.
            controller: Controller instance that owns this policy lifecycle.

        Notes:
            ``task_path`` is inferred from concrete subclass module location,
            which enables task-relative asset loading in derived policies.
        """
        self.cfg = cfg
        self.controller = controller
        # Get the module path of the actual class (works for subclasses too)
        class_module = inspect.getmodule(self.__class__)
        if class_module is None:
            raise RuntimeError("Unable to resolve policy module")
        module_file = getattr(class_module, "__file__", None)
        if module_file is None:
            raise RuntimeError("Unable to resolve policy module path")
        self.task_path = os.path.dirname(module_file)

        self._backend: str | None = None
        self._model: torch.jit.ScriptModule | None = None
        self._onnx_session: Any = None
        self._onnx_input_names: list[str] = []
        self._onnx_output_names: list[str] = []
        self._runtime_checkpoint_path: str | None = None

    @abstractmethod
    def reset(self) -> None:
        """Called when the controller starts."""

    @abstractmethod
    def inference(self) -> torch.Tensor:
        """Called each controller step to perform inference.

        Returns:
            action torch.Tensor containing the action for this step.
        """

    def resolve_checkpoint_path(self, checkpoint_path: str) -> str:
        """Resolve checkpoint path using task-local and cwd fallback."""
        path = checkpoint_path.strip()
        if not path:
            raise ValueError("checkpoint_path is empty")
        if os.path.isabs(path):
            return path

        task_relative = os.path.join(self.task_path, path)
        if os.path.exists(task_relative):
            return task_relative
        if os.path.exists(path):
            return path
        return task_relative

    def initialize_model_runtime(self, checkpoint_path: str | None = None) -> None:
        """Initialize model runtime from checkpoint path (.pt/.onnx)."""
        raw_path = checkpoint_path or self.cfg.checkpoint_path
        resolved_path = self.resolve_checkpoint_path(raw_path)
        self._runtime_checkpoint_path = resolved_path

        suffix = Path(resolved_path).suffix.lower()
        if suffix == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError(
                    "ONNX backend requested but onnxruntime is not installed"
                ) from exc

            providers = list(getattr(self.cfg, "onnx_providers", ["CPUExecutionProvider"]))
            self._onnx_session = ort.InferenceSession(
                resolved_path, providers=providers
            )
            self._onnx_input_names = [
                item.name for item in self._onnx_session.get_inputs()
            ]
            self._onnx_output_names = [
                item.name for item in self._onnx_session.get_outputs()
            ]
            self._model = None
            self._backend = "onnx"
            return

        model = torch.jit.load(resolved_path, map_location=self.cfg.device)
        try:
            model.to(self.cfg.device)
        except Exception:
            pass
        model.eval()
        self._model = cast(torch.jit.ScriptModule, model)
        self._onnx_session = None
        self._onnx_input_names = []
        self._onnx_output_names = []
        self._backend = "torch"

    def infer_model(self, obs: torch.Tensor) -> torch.Tensor:
        """Run backend inference using initialized runtime state."""
        if self._backend == "onnx":
            if self._onnx_session is None:
                raise RuntimeError("ONNX session is not initialized")
            outputs = self._onnx_session.run(
                self._onnx_output_names,
                self.prepare_onnx_inputs(obs),
            )
            return self.parse_onnx_outputs(outputs)

        if self._backend == "torch":
            if self._model is None:
                raise RuntimeError("Torch model is not initialized")
            return cast(torch.jit.ScriptModule, self._model)(obs)

        raise RuntimeError("Model runtime is not initialized")

    def prepare_onnx_inputs(self, obs: torch.Tensor) -> dict[str, Any]:
        """Build default ONNX input map for single-input models."""
        if not self._onnx_input_names:
            raise RuntimeError("ONNX inputs are not initialized")
        if len(self._onnx_input_names) != 1:
            raise RuntimeError(
                "Default prepare_onnx_inputs supports single-input ONNX models only"
            )
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)
        return {self._onnx_input_names[0]: obs_np}

    def parse_onnx_outputs(self, outputs: list[Any]) -> torch.Tensor:
        """Convert default ONNX first output to torch tensor."""
        if not outputs:
            raise RuntimeError("ONNX returned no outputs")
        output = outputs[0]
        if isinstance(output, torch.Tensor):
            tensor = output.detach()
        else:
            tensor = torch.as_tensor(output)
        return tensor.to(device=self.cfg.device, dtype=torch.float32)

    def log_named_vector(self, name: str, value: Any) -> None:
        """Log one named runtime value as summary statistics."""
        self.log_stats({name: value})

    def log_stats(self, stats: dict[str, Any]) -> None:
        """Append summary stats to CSV at ``PolicyCfg.stat_log_path``."""
        log_path = getattr(self.cfg, "stat_log_path", None)
        if not log_path:
            return

        rows: list[dict[str, Any]] = []
        step = int(getattr(self.controller, "_step_count", -1))
        time_s = self.controller.get_time() if hasattr(self.controller, "get_time") else 0.0

        for name, value in stats.items():
            kind, shape, v_min, v_max, v_mean, v_std = self._summarize_stat_value(value)
            rows.append(
                {
                    "step": step,
                    "time_s": float(time_s),
                    "name": name,
                    "kind": kind,
                    "shape": shape,
                    "min": v_min,
                    "max": v_max,
                    "mean": v_mean,
                    "std": v_std,
                }
            )

        self._append_stat_rows(log_path, rows)

    def _summarize_stat_value(
        self, value: Any
    ) -> tuple[str, str, float | None, float | None, float | None, float | None]:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
            kind = "tensor"
        elif isinstance(value, np.ndarray):
            array = value
            kind = "ndarray"
        elif isinstance(value, (list, tuple)):
            array = np.asarray(value)
            kind = "sequence"
        elif isinstance(value, (int, float)):
            scalar = float(value)
            return "scalar", "()", scalar, scalar, scalar, 0.0
        else:
            return type(value).__name__, "", None, None, None, None

        if array.size == 0:
            return kind, str(list(array.shape)), None, None, None, None

        array = array.astype(np.float32, copy=False)
        return (
            kind,
            str(list(array.shape)),
            float(array.min()),
            float(array.max()),
            float(array.mean()),
            float(array.std()),
        )

    def _append_stat_rows(self, log_path: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = Path(log_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "step",
                "time_s",
                "name",
                "kind",
                "shape",
                "min",
                "max",
                "mean",
                "std",
            ]
            file_exists = path.exists() and path.stat().st_size > 0
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(rows)
        except Exception:
            # Best-effort runtime logging: never break control loop inference.
            return


class BaseController:
    """Simple deployment environment skeleton and execution overview.

    This class provides a minimal, dependency-light interface suitable for
    deployment scripts and controllers. It defines the method contract used by
    concrete controller implementations and documents the typical runtime
    execution order.

    Public method contract
    - `start(initial_state=None) -> obs`: prepare controller and policy for
        execution and return initial observation.
    - `policy_step() -> torch.Tensor`: invoke policy inference for one step
        and return the action tensor.
    - `ctrl_step(dof_targets: torch.Tensor) -> None`: apply action to the
        environment (send to actuators / shared buffer / simulator).
    - `update_state() -> None`: refresh internal robot state from sensors or
        shared buffers (called each control loop iteration before inference).
    - `stop() -> None`: stop the running session; should be idempotent.
    - `run() -> None`: high-level entry point for a controller process or
        thread (optional to implement for each concrete controller).

    Concrete controllers may implement `run()` to orchestrate the typical
    execution flow below:

        start()

            |
            v
    +----------------- main loop -----------------+
    |  update_state()                                |
    |      |                                         |
    |      v                                         |
    |  policy_step()  -> (action tensor)            |
    |      |                                         |
    |      v                                         |
    |  ctrl_step(action)                             |
    |      |                                         |
    +-----------------------------------------------+
            |
            v
            stop() -> cleanup()/finalize()

    Notes and recommendations
    - `update_state()` should read the latest sensor/shared-buffer data and
        populate `self.robot.data` before `policy_step()` is called.
    - `policy_step()` is responsible only for producing actions and should
        not have side-effects that interfere with `update_state()`.
    - `ctrl_step()` applies the action produced by the policy to actuators or
        publish it.
    """

    cfg: ControllerCfg
    robot: BoosterRobot
    vel_command: VelocityCommand | None
    policy: Policy

    def __init__(self, cfg: ControllerCfg) -> None:
        """Initialize controller runtime objects from ``ControllerCfg``.

        Args:
            cfg: Unified controller configuration (robot, policy, timing, IO).

        Notes:
            This constructor initializes robot model wrappers, optional velocity
            command interfaces, and policy instance via configured constructor.
        """
        self.cfg = cfg
        self._step_count: int = 0
        self._elapsed_s: float = 0.0
        self.is_running: bool = False
        self.robot = BoosterRobot(cfg.robot)
        self.vel_command = None
        vel_cfg = self.cfg.vel_command
        if vel_cfg is not None:
            self.vel_command = VelocityCommand(
                cast(VelocityCommandCfg, vel_cfg)
            )
        self.policy = self.cfg.policy.constructor(self.cfg.policy, self)
    
    def get_time(self) -> float:
        """Get elapsed time since start of current session in seconds."""
        return self._elapsed_s

    def start(self):
        """Begin a deployment session.

        This method resets runtime counters and invokes ``policy.reset()``.
        It must be called before ``policy_step()``.
        """
        self._step_count = 0
        self._elapsed_s = 0.0
        self.is_running = True
        self.policy.reset()

    def policy_step(self) -> torch.Tensor:
        """Execute one inference step and return the action.

        Returns:
            action tensor
        """
        if not self.is_running:
            raise RuntimeError("Environment.step() called before start().")

        self._step_count += 1
        self._elapsed_s = self._step_count * self.cfg.policy_dt

        return self.policy.inference()

    def stop(self) -> None:
        """Mark deployment session as stopped.

        Concrete controllers may extend this to trigger backend-specific
        shutdown signals. This base implementation is intentionally idempotent.
        """
        self.is_running = False

    @abstractmethod
    def ctrl_step(self, dof_targets: torch.Tensor) -> None:
        """Advance the environment by one control step.

        Args:
            dof_targets: Action tensor for this step (dof targets).
        """

    @abstractmethod
    def update_state(self) -> None:
        """Update robot data from sensors or shared buffers."""

    @abstractmethod
    def run(self) -> None:
        """Main loop entry point."""
