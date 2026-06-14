# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import yaml

from legged_lab.assets.magicbot import (
    MAGICBOT_Z1_BIG_DAMPING,
    MAGICBOT_Z1_BIG_STIFFNESS,
    MAGICBOT_Z1_SMALL_DAMPING,
    MAGICBOT_Z1_SMALL_STIFFNESS,
)


DEFAULT_DEPLOY_ROOT = Path("/home/hiyio/MaigcLab/RoboMimic_Deploy_magicbot")
CONFIG_FILENAMES = ("LocoMode_lowKp.yaml", "LocoMode.yaml")

# Policy/action order used by the legacy Python deploy and the current C++ deploy.
LAB_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "head_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

BIG_JOINT_NAMES = {
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
}

JOINT2MOTOR_IDX = [
    0,
    6,
    12,
    1,
    7,
    13,
    14,
    19,
    2,
    8,
    15,
    20,
    3,
    9,
    16,
    21,
    4,
    10,
    17,
    22,
    5,
    11,
    18,
    23,
]

DEFAULT_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.15,
    "right_shoulder_pitch_joint": 0.15,
    "left_shoulder_roll_joint": 0.15,
    "right_shoulder_roll_joint": -0.15,
    "left_knee_joint": 0.35,
    "right_knee_joint": 0.35,
    "left_ankle_pitch_joint": -0.18,
    "right_ankle_pitch_joint": -0.18,
    "left_elbow_joint": 0.5,
    "right_elbow_joint": 0.5,
}


def is_magicbot_z1_task(task_name: str | None) -> bool:
    return bool(task_name and task_name.startswith("magicbot_z1"))


def build_magicbot_z1_deploy_yaml(env_cfg) -> dict:
    num_actions = len(LAB_JOINT_NAMES)
    command_dim = 4
    gait_phase_dim = 0
    num_obs = 6 + command_dim + 3 * num_actions + gait_phase_dim
    root_height_command = _root_height_command(env_cfg)
    command_scale = float(getattr(env_cfg.normalization.obs_scales, "commands", 1.0))

    _validate_deploy_supported(env_cfg)

    return {
        "policy_path": "policy.onnx",
        "command_dim": command_dim,
        "num_actions": num_actions,
        "num_obs": num_obs,
        "gait_phase_dim": gait_phase_dim,
        "gait_phase_period": 0.6,
        "gait_phase_stand_threshold": 0.02,
        "policy_dt": _policy_dt(env_cfg),
        "joint2motor_idx": JOINT2MOTOR_IDX,
        "kps": _joint_family_values(MAGICBOT_Z1_BIG_STIFFNESS, MAGICBOT_Z1_SMALL_STIFFNESS),
        "kds": _joint_family_values(MAGICBOT_Z1_BIG_DAMPING, MAGICBOT_Z1_SMALL_DAMPING),
        "tau_limit_scale": 1,
        "tau_limit": _joint_family_values(120, 50),
        "default_angles": [float(DEFAULT_JOINT_POS.get(name, 0.0)) for name in LAB_JOINT_NAMES],
        "cmd_scale": [command_scale] * command_dim,
        "root_height_command": root_height_command,
        "cmd_deadzone": 0.0,
        "cmd_slew_rate": _list_floats(getattr(env_cfg.commands, "command_slew_rate", (0.0, 0.0, 0.0))),
        "cmd_init": [0.0, 0.0, 0.0, root_height_command],
        "cmd_range": {
            "lin_vel_x": _range_list(env_cfg.commands.ranges.lin_vel_x),
            "lin_vel_y": _range_list(env_cfg.commands.ranges.lin_vel_y),
            "ang_vel_z": _range_list(env_cfg.commands.ranges.ang_vel_z),
        },
        "ang_vel_scale": float(env_cfg.normalization.obs_scales.ang_vel),
        "dof_pos_scale": float(env_cfg.normalization.obs_scales.joint_pos),
        "dof_vel_scale": float(env_cfg.normalization.obs_scales.joint_vel),
        "obs_clip": float(env_cfg.normalization.clip_observations),
        "action_scale": float(env_cfg.robot.action_scale),
    }


def export_magicbot_z1_deploy_yamls(
    task_name: str | None,
    env_cfg,
    log_dir: str | os.PathLike[str] | None = None,
    deploy_root: str | os.PathLike[str] | None = None,
    extra_dirs: Iterable[str | os.PathLike[str]] = (),
) -> list[Path]:
    if not is_magicbot_z1_task(task_name):
        return []

    cfg = build_magicbot_z1_deploy_yaml(env_cfg)
    output_dirs: list[Path] = []
    if log_dir is not None:
        output_dirs.append(Path(log_dir) / "deploy")

    root = (
        Path(deploy_root)
        if deploy_root is not None
        else Path(os.environ.get("MAGICBOT_DEPLOY_ROOT", DEFAULT_DEPLOY_ROOT))
    )
    if root:
        output_dirs.append(root / "policies" / "loco_mode" / "config")

    output_dirs.extend(Path(p) for p in extra_dirs)

    written: list[Path] = []
    for out_dir in output_dirs:
        for filename in CONFIG_FILENAMES:
            path = out_dir / filename
            _write_yaml(path, cfg)
            written.append(path)
    return written


def _validate_deploy_supported(env_cfg) -> None:
    if int(env_cfg.robot.actor_obs_history_length) != 1:
        raise ValueError("MagicBot Z1 deploy YAML supports actor_obs_history_length == 1 only")
    if bool(env_cfg.scene.height_scanner.enable_height_scan):
        raise ValueError("MagicBot Z1 deploy YAML does not support height_scan observations")


def _joint_family_values(big_value: float, small_value: float) -> list[float]:
    return [float(big_value if name in BIG_JOINT_NAMES else small_value) for name in LAB_JOINT_NAMES]


def _policy_dt(env_cfg) -> float:
    return float(env_cfg.sim.dt) * float(env_cfg.sim.decimation)


def _root_height_command(env_cfg) -> float:
    root_height = float(env_cfg.commands.root_height)
    if root_height > 0.0:
        return root_height
    return float(env_cfg.scene.robot.init_state.pos[2])


def _range_list(values) -> list[float]:
    return [float(values[0]), float(values[1])]


def _list_floats(values) -> list[float]:
    return [float(value) for value in values]


def _write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    path.write_text(text, encoding="utf-8")
