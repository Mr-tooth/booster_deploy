"""Tests for the T1 getup deployment policy behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from booster_deploy.controllers.base_controller import BoosterRobot
from booster_deploy.robots.booster import T1_23DOF_CFG
from tasks.t1_getup.t1_getup import T1GetupPolicy


class _DummyController:
    """Minimal controller stub used by policy unit tests."""

    def __init__(self) -> None:
        """Initialize controller stub fields used by policy runtime."""
        self.robot = BoosterRobot(T1_23DOF_CFG)
        self._step_count = 0
        self._stopped = False

    def get_time(self) -> float:
        """Return a deterministic controller time for tests."""
        return self._step_count * 0.02

    def stop(self) -> None:
        """Record that stop was requested by the policy."""
        self._stopped = True


def _make_cfg(
    checkpoint_path: str,
    *,
    settle_steps: int = 50,
    action_scale: float = 0.6,
) -> SimpleNamespace:
    """Build a minimal policy config namespace for tests."""
    return SimpleNamespace(
        checkpoint_path=checkpoint_path,
        device="cpu",
        enable_safety_fallback=False,
        stat_log_path=None,
        stat_log_inference_io=False,
        onnx_providers=["CPUExecutionProvider"],
        policy_joint_names=list(T1_23DOF_CFG.joint_names),
        action_scale=action_scale,
        settle_steps=settle_steps,
        clip_obs=100.0,
        clip_action=100.0,
        obs_ang_vel_scale=1.0,
        obs_dof_pos_scale=1.0,
        obs_dof_vel_scale=1.0,
        obs_last_action_scale=1.0,
    )


def _save_constant_actor(path: Path) -> None:
    """Save a scripted actor that always outputs ones with 23 dims."""

    class _ConstantActor(torch.nn.Module):
        """Tiny actor model for deterministic policy tests."""

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            """Return a constant action tensor for each batch row."""
            batch_size = obs.shape[0]
            return torch.ones(batch_size, 23, dtype=torch.float32)

    model = torch.jit.script(_ConstantActor())
    model.save(str(path))


def _set_robot_state(controller: _DummyController) -> None:
    """Populate deterministic robot state tensors for policy tests."""
    robot_data = controller.robot.data
    robot_data.root_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    robot_data.root_ang_vel_b = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    robot_data.joint_pos = torch.linspace(-0.22, 0.22, 23, dtype=torch.float32)
    robot_data.joint_vel = torch.linspace(0.4, -0.4, 23, dtype=torch.float32)


class TestT1GetupPolicy(unittest.TestCase):
    """Validate T1 getup observation and relative action semantics."""

    def test_compute_observation_shape(self) -> None:
        """Observation should match actor input dimension (75)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "actor.pt"
            _save_constant_actor(ckpt_path)

            controller = _DummyController()
            _set_robot_state(controller)
            policy = T1GetupPolicy(_make_cfg(str(ckpt_path)), controller)

            obs = policy.compute_observation()
            self.assertEqual(tuple(obs.shape), (75,))

    def test_inference_holds_current_pose_in_settle_window(self) -> None:
        """During settle window, targets should hold current joint position."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "actor.pt"
            _save_constant_actor(ckpt_path)

            controller = _DummyController()
            _set_robot_state(controller)
            controller._step_count = 1
            policy = T1GetupPolicy(_make_cfg(str(ckpt_path), settle_steps=50), controller)

            current_pos = controller.robot.data.joint_pos.clone()
            dof_targets = policy.inference()

            self.assertTrue(torch.allclose(dof_targets, current_pos))
            self.assertTrue(torch.allclose(policy.last_action, torch.ones(23, dtype=torch.float32)))

    def test_inference_applies_relative_delta_after_settle(self) -> None:
        """After settle window, targets should be current + action*scale."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "actor.pt"
            _save_constant_actor(ckpt_path)

            controller = _DummyController()
            _set_robot_state(controller)
            controller._step_count = 100
            policy = T1GetupPolicy(_make_cfg(str(ckpt_path), settle_steps=50, action_scale=0.6), controller)

            current_pos = controller.robot.data.joint_pos.clone()
            dof_targets = policy.inference()
            expected = current_pos + 0.6

            self.assertTrue(torch.allclose(dof_targets, expected))


if __name__ == "__main__":
    unittest.main()
