# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.

from __future__ import annotations

import os
from collections.abc import Sequence

import isaaclab.utils.math as math_utils
import numpy as np
import torch


def _npz_name_list(data: np.lib.npyio.NpzFile, key: str) -> list[str] | None:
    if key not in data:
        return None
    values = np.asarray(data[key])
    if values.ndim == 0:
        return [str(values.item())]
    return [str(value) for value in values.tolist()]


def _remap_named_tensor(
    tensor: torch.Tensor,
    source_names: Sequence[str] | None,
    target_names: Sequence[str],
    tensor_name: str,
    motion_file: str,
) -> torch.Tensor:
    if source_names is None:
        if tensor.shape[1] != len(target_names):
            raise ValueError(
                f"{tensor_name} in {motion_file} has dim {tensor.shape[1]}, "
                f"but {len(target_names)} target names were requested and no name metadata exists."
            )
        return tensor

    if tensor.shape[1] != len(source_names):
        raise ValueError(
            f"{tensor_name} metadata mismatch in {motion_file}: "
            f"{len(source_names)} names vs dim {tensor.shape[1]}."
        )

    index_by_name = {name: idx for idx, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in index_by_name]
    if missing:
        raise ValueError(f"{tensor_name} in {motion_file} is missing required names: {missing}.")

    selected = [index_by_name[name] for name in target_names]
    return tensor[:, selected]


def _fill_robot_body_tensor(
    tensor: torch.Tensor,
    source_names: Sequence[str] | None,
    robot_body_names: Sequence[str],
    required_body_names: Sequence[str],
    tensor_name: str,
    motion_file: str,
) -> torch.Tensor:
    if source_names is None:
        if tensor.shape[1] != len(robot_body_names):
            raise ValueError(
                f"{tensor_name} in {motion_file} has dim {tensor.shape[1]}, "
                f"but robot has {len(robot_body_names)} bodies and no name metadata exists."
            )
        return tensor

    if tensor.shape[1] != len(source_names):
        raise ValueError(
            f"{tensor_name} metadata mismatch in {motion_file}: "
            f"{len(source_names)} names vs dim {tensor.shape[1]}."
        )

    source_index_by_name = {name: idx for idx, name in enumerate(source_names)}
    missing_required = [name for name in required_body_names if name not in source_index_by_name]
    if missing_required:
        raise ValueError(f"{tensor_name} in {motion_file} is missing required bodies: {missing_required}.")

    output = torch.zeros(
        (tensor.shape[0], len(robot_body_names)) + tuple(tensor.shape[2:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    for robot_body_id, name in enumerate(robot_body_names):
        source_id = source_index_by_name.get(name)
        if source_id is not None:
            output[:, robot_body_id] = tensor[:, source_id]
    return output


def _quaternion_to_tangent_and_normal(quat_wxyz: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(quat_wxyz[..., :3])
    ref_normal = torch.zeros_like(quat_wxyz[..., :3])
    ref_tangent[..., 0] = 1.0
    ref_normal[..., 2] = 1.0
    tangent = math_utils.quat_apply(quat_wxyz, ref_tangent)
    normal = math_utils.quat_apply(quat_wxyz, ref_normal)
    return torch.cat((tangent, normal), dim=-1)


class ReferenceMotion:
    """Name-aware reference motion sampler for AMP and light style rewards."""

    def __init__(
        self,
        motion_file: str,
        robot_joint_names: Sequence[str],
        robot_body_names: Sequence[str],
        num_envs: int,
        device: str,
        anchor_body_name: str,
        amp_key_body_names: Sequence[str],
        reward_body_names: Sequence[str],
        min_command_speed: float,
        max_reference_speed: float,
        speed_match_tolerance: float,
        speed_sample_jitter_frames: int,
        amp_observation_history_length: int,
        command_conditioned_sampling: bool = False,
        command_sample_candidates: int = 1,
        command_lin_vel_x_scale: float = 0.75,
        command_lin_vel_y_scale: float = 0.35,
        command_yaw_scale: float = 0.45,
    ) -> None:
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Invalid reference motion file: {motion_file}")

        self.motion_file = motion_file
        self.device = device
        self.robot_joint_names = list(robot_joint_names)
        self.robot_body_names = list(robot_body_names)
        self.anchor_body_name = anchor_body_name
        self.anchor_body_id = self.robot_body_names.index(anchor_body_name)
        self.amp_key_body_names = list(amp_key_body_names)
        self.amp_key_body_ids = torch.tensor(
            [self.robot_body_names.index(name) for name in self.amp_key_body_names],
            dtype=torch.long,
            device=device,
        )
        self.reward_body_names = list(reward_body_names)
        self.reward_body_ids = torch.tensor(
            [self.robot_body_names.index(name) for name in self.reward_body_names],
            dtype=torch.long,
            device=device,
        )
        self.min_command_speed = float(min_command_speed)
        self.max_reference_speed = float(max_reference_speed)
        self.speed_match_tolerance = float(speed_match_tolerance)
        self.speed_sample_jitter_frames = int(speed_sample_jitter_frames)
        self.amp_observation_history_length = int(amp_observation_history_length)
        self.command_conditioned_sampling = bool(command_conditioned_sampling)
        self.command_sample_candidates = max(1, int(command_sample_candidates))
        self.command_lin_vel_x_scale = max(float(command_lin_vel_x_scale), 1.0e-6)
        self.command_lin_vel_y_scale = max(float(command_lin_vel_y_scale), 1.0e-6)
        self.command_yaw_scale = max(float(command_yaw_scale), 1.0e-6)

        required_body_names = sorted(set([anchor_body_name, *self.amp_key_body_names, *self.reward_body_names]))
        with np.load(motion_file, allow_pickle=True) as data:
            fps = data["fps"]
            if isinstance(fps, np.ndarray):
                fps = fps.item() if fps.size == 1 else float(fps.flatten()[0])
            self.fps = float(fps)

            motion_joint_names = _npz_name_list(data, "joint_names")
            motion_body_names = _npz_name_list(data, "body_names")

            required_keys = (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            )
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValueError(f"Reference motion {motion_file} is missing arrays: {missing_keys}.")

            joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            self.joint_pos = _remap_named_tensor(
                joint_pos, motion_joint_names, self.robot_joint_names, "joint_pos", motion_file
            )
            self.joint_vel = _remap_named_tensor(
                joint_vel, motion_joint_names, self.robot_joint_names, "joint_vel", motion_file
            )

            self.body_pos_w = _fill_robot_body_tensor(
                torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device),
                motion_body_names,
                self.robot_body_names,
                required_body_names,
                "body_pos_w",
                motion_file,
            )
            self.body_quat_w = _fill_robot_body_tensor(
                torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device),
                motion_body_names,
                self.robot_body_names,
                required_body_names,
                "body_quat_w",
                motion_file,
            )
            self.body_lin_vel_w = _fill_robot_body_tensor(
                torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device),
                motion_body_names,
                self.robot_body_names,
                required_body_names,
                "body_lin_vel_w",
                motion_file,
            )
            self.body_ang_vel_w = _fill_robot_body_tensor(
                torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device),
                motion_body_names,
                self.robot_body_names,
                required_body_names,
                "body_ang_vel_w",
                motion_file,
            )

        self.num_frames = int(self.joint_pos.shape[0])
        self.dt = 1.0 / self.fps
        self.frame_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.anchor_lin_vel_yaw = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.body_quat_w[:, self.anchor_body_id]),
            self.body_lin_vel_w[:, self.anchor_body_id],
        )
        self.anchor_yaw_rate = self.body_ang_vel_w[:, self.anchor_body_id, 2]
        self.anchor_speed = torch.norm(self.anchor_lin_vel_yaw[:, :2], dim=-1)
        eligible = self.anchor_speed >= self.min_command_speed
        if self.max_reference_speed > 0.0:
            eligible &= self.anchor_speed <= self.max_reference_speed
        if not bool(torch.any(eligible).item()):
            eligible = torch.ones_like(self.anchor_speed, dtype=torch.bool)
        self.eligible_frame_ids = torch.where(eligible)[0]
        sorted_order = torch.argsort(self.anchor_speed[self.eligible_frame_ids])
        self.sorted_frame_ids = self.eligible_frame_ids[sorted_order]
        self.sorted_speeds = self.anchor_speed[self.sorted_frame_ids]

    @property
    def amp_observation_dim(self) -> int:
        return 2 * len(self.robot_joint_names) + 13 + 3 * len(self.amp_key_body_names)

    def reset(
        self,
        env_ids: torch.Tensor,
        command_xy: torch.Tensor | None = None,
        command_values: torch.Tensor | None = None,
    ) -> None:
        if env_ids.numel() == 0:
            return
        if self.command_conditioned_sampling and command_values is not None:
            self.frame_ids[env_ids] = self.sample_frame_ids_by_command(command_values[env_ids])
            return
        if command_xy is None:
            random_ids = torch.randint(
                0, self.eligible_frame_ids.numel(), (env_ids.numel(),), device=self.device
            )
            self.frame_ids[env_ids] = self.eligible_frame_ids[random_ids]
            return
        command_speed = torch.norm(command_xy[env_ids], dim=-1)
        self.frame_ids[env_ids] = self.sample_frame_ids_by_speed(command_speed)

    def update(
        self,
        command_xy: torch.Tensor,
        step_dt: float,
        command_values: torch.Tensor | None = None,
    ) -> None:
        if self.frame_ids.numel() == 0:
            return
        advance = max(1, int(round(float(step_dt) * self.fps)))
        self.frame_ids.add_(advance).remainder_(self.num_frames)

        command_speed = torch.norm(command_xy, dim=-1)
        target_speed = self._clamp_to_reference_speed(command_speed)
        ref_speed = self.anchor_speed[self.frame_ids]
        should_match = command_speed >= self.min_command_speed
        mismatch = torch.abs(ref_speed - target_speed) > self.speed_match_tolerance
        env_ids = torch.where(should_match & mismatch)[0]
        if env_ids.numel() > 0:
            if self.command_conditioned_sampling and command_values is not None:
                self.frame_ids[env_ids] = self.sample_frame_ids_by_command(command_values[env_ids])
            else:
                self.frame_ids[env_ids] = self.sample_frame_ids_by_speed(command_speed[env_ids])

    def sample_frame_ids_by_speed(self, command_speed: torch.Tensor) -> torch.Tensor:
        target_speed = self._clamp_to_reference_speed(command_speed)
        positions = torch.searchsorted(self.sorted_speeds.contiguous(), target_speed).clamp(
            min=0, max=max(int(self.sorted_speeds.numel()) - 1, 0)
        )
        if self.speed_sample_jitter_frames > 0 and positions.numel() > 0:
            jitter = torch.randint(
                -self.speed_sample_jitter_frames,
                self.speed_sample_jitter_frames + 1,
                positions.shape,
                device=self.device,
            )
            positions = (positions + jitter).clamp(min=0, max=int(self.sorted_speeds.numel()) - 1)
        return self.sorted_frame_ids[positions]

    def sample_frame_ids_by_command(self, command_values: torch.Tensor) -> torch.Tensor:
        commands = command_values.to(self.device)
        if commands.ndim != 2 or commands.shape[1] < 2:
            raise ValueError(f"Command-conditioned sampling expects shape (N, >=2), got {tuple(commands.shape)}.")
        target_xy = commands[:, :2]
        target_speed = self._clamp_to_reference_speed(torch.norm(target_xy, dim=-1))
        base_positions = torch.searchsorted(self.sorted_speeds.contiguous(), target_speed).clamp(
            min=0, max=max(int(self.sorted_speeds.numel()) - 1, 0)
        )

        num_candidates = self.command_sample_candidates
        if num_candidates <= 1:
            return self.sorted_frame_ids[base_positions]

        jitter_width = max(self.speed_sample_jitter_frames, 1)
        offsets = torch.randint(
            -jitter_width,
            jitter_width + 1,
            (commands.shape[0], num_candidates),
            device=self.device,
        )
        offsets[:, 0] = 0
        positions = (base_positions.unsqueeze(1) + offsets).clamp(min=0, max=int(self.sorted_speeds.numel()) - 1)
        candidate_ids = self.sorted_frame_ids[positions]

        ref_xy = self.anchor_lin_vel_yaw[candidate_ids, :2]
        ref_yaw = self.anchor_yaw_rate[candidate_ids]
        command_yaw = commands[:, 2].unsqueeze(1) if commands.shape[1] >= 3 else torch.zeros_like(ref_yaw)

        speed_error = (self.anchor_speed[candidate_ids] - target_speed.unsqueeze(1)) / max(
            self.speed_match_tolerance, 1.0e-6
        )
        x_error = (ref_xy[..., 0] - target_xy[:, 0].unsqueeze(1)) / self.command_lin_vel_x_scale
        y_error = (ref_xy[..., 1] - target_xy[:, 1].unsqueeze(1)) / self.command_lin_vel_y_scale
        yaw_error = (ref_yaw - command_yaw) / self.command_yaw_scale
        score = speed_error.square() + x_error.square() + y_error.square() + yaw_error.square()
        best_candidate = torch.argmin(score, dim=1)
        return candidate_ids[torch.arange(commands.shape[0], device=self.device), best_candidate]

    def sample_expert_amp_observations(
        self,
        num_samples: int,
        num_frames: int | None = None,
        command_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frame_count = self.amp_observation_history_length if num_frames is None else int(num_frames)
        if self.command_conditioned_sampling and command_values is not None:
            if int(command_values.shape[0]) != int(num_samples):
                raise ValueError(
                    "Command-conditioned expert sampling expects one command per sample, "
                    f"got {tuple(command_values.shape)} for {num_samples} samples."
                )
            base_frame_ids = self.sample_frame_ids_by_command(command_values)
        else:
            random_ids = torch.randint(0, self.eligible_frame_ids.numel(), (num_samples,), device=self.device)
            base_frame_ids = self.eligible_frame_ids[random_ids]
        observations = []
        for frame_offset in range(frame_count):
            frame_ids = (base_frame_ids - frame_offset).remainder(self.num_frames)
            observations.append(self.build_amp_observations_from_frames(frame_ids))
        return torch.stack(observations, dim=1)

    def build_amp_observations_from_frames(self, frame_ids: torch.Tensor) -> torch.Tensor:
        return self.build_amp_observations(
            self.joint_pos[frame_ids],
            self.joint_vel[frame_ids],
            self.body_pos_w[frame_ids],
            self.body_quat_w[frame_ids],
            self.body_lin_vel_w[frame_ids],
            self.body_ang_vel_w[frame_ids],
        )

    def build_amp_observations(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        body_ang_vel_w: torch.Tensor,
    ) -> torch.Tensor:
        anchor_pos = body_pos_w[:, self.anchor_body_id]
        anchor_quat = body_quat_w[:, self.anchor_body_id]
        key_body_pos = body_pos_w[:, self.amp_key_body_ids]
        key_body_pos_rel = key_body_pos - anchor_pos.unsqueeze(1)
        return torch.cat(
            (
                joint_pos,
                joint_vel,
                anchor_pos[:, 2:3],
                _quaternion_to_tangent_and_normal(anchor_quat),
                body_lin_vel_w[:, self.anchor_body_id],
                body_ang_vel_w[:, self.anchor_body_id],
                key_body_pos_rel.reshape(key_body_pos_rel.shape[0], -1),
            ),
            dim=-1,
        )

    @property
    def current_joint_pos(self) -> torch.Tensor:
        return self.joint_pos[self.frame_ids]

    @property
    def current_joint_vel(self) -> torch.Tensor:
        return self.joint_vel[self.frame_ids]

    @property
    def current_body_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[self.frame_ids]

    @property
    def current_body_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[self.frame_ids]

    def _clamp_to_reference_speed(self, speed: torch.Tensor) -> torch.Tensor:
        return speed.clamp(min=float(self.sorted_speeds[0].item()), max=float(self.sorted_speeds[-1].item()))
