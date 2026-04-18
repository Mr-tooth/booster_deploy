"""Load motion datasets and expose joint/body trajectories as torch tensors."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch

from booster_deploy.utils.isaaclab import math as lab_math


class MotionLoader:
    """Load a motion `.npz` file with optional joint/body subset remapping."""

    def __init__(
        self,
        motion_file: str,
        track_body_names: Sequence[str] | None = None,
        track_joint_names: Sequence[str] | None = None,
        *,
        default_motion_body_names: Sequence[str] | None = None,
        default_motion_joint_names: Sequence[str] | None = None,
        align_to_first_frame: bool = False,
        device: str = "cpu",
    ) -> None:
        """Load motion data and prepare tensors on the target device.

        Args:
            motion_file: Path to the motion `.npz` file.
            track_body_names: Optional ordered subset of body names to expose.
            track_joint_names: Optional ordered subset of joint names to expose.
            default_motion_body_names: Fallback body-name list when absent in file.
            default_motion_joint_names: Fallback joint-name list when absent in file.
            align_to_first_frame: Whether to remove initial root translation/yaw.
            device: Torch device name for loaded tensors.

        Raises:
            AssertionError: If `motion_file` does not exist or fallback names are
                required but missing.

        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self.device: str | torch.device = device
        data = np.load(motion_file)
        self.fps: np.ndarray = data["fps"]

        self._body_names = self._resolve_names(
            names_from_file=data["body_names"].tolist() if "body_names" in data else None,
            requested_names=track_body_names,
            default_names=default_motion_body_names,
            kind="body",
            motion_file=motion_file,
        )
        self._joint_names = self._resolve_names(
            names_from_file=data["joint_names"].tolist() if "joint_names" in data else None,
            requested_names=track_joint_names,
            default_names=default_motion_joint_names,
            kind="joint",
            motion_file=motion_file,
        )

        self.track_body_names = list(track_body_names) if track_body_names is not None else self._body_names
        self.track_joint_names = list(track_joint_names) if track_joint_names is not None else self._joint_names

        self._body_indexes = self._resolve_track_indexes(
            all_names=self._body_names,
            track_names=self.track_body_names,
            length_hint=int(data["body_pos_w"].shape[1]),
            device=device,
        )
        self._joint_indexes = self._resolve_track_indexes(
            all_names=self._joint_names,
            track_names=self.track_joint_names,
            length_hint=int(data["joint_pos"].shape[1]),
            device=device,
        )

        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)[:, self._joint_indexes]
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)[:, self._joint_indexes]
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

        if align_to_first_frame:
            self._align_root_to_first_frame()

        self.time_step_total = self.joint_pos.shape[0]

    @staticmethod
    def _resolve_names(
        *,
        names_from_file: list[str] | None,
        requested_names: Sequence[str] | None,
        default_names: Sequence[str] | None,
        kind: str,
        motion_file: str,
    ) -> list[str]:
        """Resolve canonical name lists from file or fallback defaults.

        Args:
            names_from_file: Names embedded in the motion file.
            requested_names: Names requested by caller for tracking.
            default_names: Default names provided by caller when file lacks names.
            kind: Name category (`"body"` or `"joint"`).
            motion_file: Motion file path for diagnostics.

        Returns:
            Canonical full name list for the chosen category.

        Raises:
            AssertionError: If names are required but unavailable.

        """
        if names_from_file is not None:
            return names_from_file
        assert requested_names is None or default_names is not None, (
            f"Motion file {motion_file} missing {kind}_names and no fallback "
            f"default_motion_{kind}_names is provided."
        )
        return list(default_names) if default_names is not None else []

    @staticmethod
    def _resolve_track_indexes(
        *,
        all_names: list[str],
        track_names: list[str],
        length_hint: int,
        device: str,
    ) -> torch.Tensor:
        """Map tracked names into tensor indexes.

        Args:
            all_names: Full canonical names in motion-file order.
            track_names: Requested tracked names in output order.
            length_hint: Full sequence length when no names are provided.
            device: Torch device for returned tensor.

        Returns:
            Long tensor of selected indexes.

        """
        if not track_names:
            return torch.arange(length_hint, dtype=torch.long, device=device)
        return torch.tensor([all_names.index(name) for name in track_names], dtype=torch.long, device=device)

    def _align_root_to_first_frame(self) -> None:
        """Align trajectory to the first frame root pose around the ground plane."""
        init_root_pos_xy = self._body_pos_w[:1, :1].clone()
        init_root_pos_xy[:, :, 2] = 0.0
        init_root_quat_yaw = lab_math.yaw_quat(self._body_quat_w[:1, :1])
        self._body_pos_w, self._body_quat_w = lab_math.subtract_frame_transforms(
            init_root_pos_xy,
            init_root_quat_yaw.repeat(*self._body_quat_w.shape[:2], 1),
            t02=self._body_pos_w,
            q02=self._body_quat_w,
        )

        q_inv = lab_math.quat_inv(init_root_quat_yaw)
        self._body_lin_vel_w = lab_math.quat_apply(q_inv, self._body_lin_vel_w)
        self._body_ang_vel_w = lab_math.quat_apply(q_inv, self._body_ang_vel_w)

    def to(self, device: str | torch.device) -> None:
        """Move all tensors and index buffers to a target device.

        Args:
            device: Target torch device.

        """
        self.device = device
        self.joint_pos = self.joint_pos.to(device)
        self.joint_vel = self.joint_vel.to(device)
        self._body_pos_w = self._body_pos_w.to(device)
        self._body_quat_w = self._body_quat_w.to(device)
        self._body_lin_vel_w = self._body_lin_vel_w.to(device)
        self._body_ang_vel_w = self._body_ang_vel_w.to(device)
        self._body_indexes = self._body_indexes.to(device)
        self._joint_indexes = self._joint_indexes.to(device)

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Return world-frame body positions for tracked bodies."""
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Return world-frame body quaternions for tracked bodies."""
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Return world-frame body linear velocities for tracked bodies."""
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Return world-frame body angular velocities for tracked bodies."""
        return self._body_ang_vel_w[:, self._body_indexes]
