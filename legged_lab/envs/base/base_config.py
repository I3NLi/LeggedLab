# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import math
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp


@configclass
class RewardCfg:
    pass


@configclass
class HeightScannerCfg:
    enable_height_scan: bool = False
    prim_body_name: str = MISSING
    resolution: float = 0.1
    size: tuple = (1.6, 1.0)
    debug_vis: bool = False
    drift_range: tuple = (0.0, 0.0)


@configclass
class BaseSceneCfg:
    max_episode_length_s: float = 20.0
    num_envs: int = 4096
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    terrain_type: str = MISSING
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5
    height_scanner: HeightScannerCfg = HeightScannerCfg()
    out_of_bounds_teleport_enable: bool = True
    out_of_bounds_teleport_trigger_scale: float = 1.5
    out_of_bounds_teleport_margin: float = -100.0


@configclass
class RobotCfg:
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_scale: float = 0.25
    terminate_contacts_body_names: list = []
    immediate_terminate_contacts_body_names: list = []
    feet_body_names: list = []
    # Allow brief recovery after a fall before terminating the episode.
    terminate_contacts_delay_s: float = 1.0
    terminate_when_stuck: bool = False
    stuck_command_threshold: float = 0.2
    stuck_speed_threshold: float = 0.08
    stuck_grace_s: float = 1.0
    stuck_duration_s: float = 0.6
    terminate_when_speed_tracking_failed: bool = False
    speed_tracking_command_threshold: float = 0.5
    speed_tracking_abs_error_threshold: float = 0.5
    speed_tracking_rel_error_threshold: float = 0.35
    speed_tracking_grace_s: float = 2.0
    speed_tracking_duration_s: float = 1.2


@configclass
class ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    projected_gravity: float = 1.0
    commands: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actions: float = 1.0
    height_scan: float = 1.0


@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    height_scan_offset: float = 0.5


@configclass
class CommandRangesCfg:
    lin_vel_x: tuple = (-0.6, 1.0)
    lin_vel_y: tuple = (-0.5, 0.5)
    ang_vel_z: tuple = (-1.0, 1.0)
    heading: tuple = (-math.pi, math.pi)


@configclass
class CommandsCfg:
    resampling_time_range: tuple = (10.0, 10.0)
    rel_standing_envs: float = 0.2
    rel_heading_envs: float = 1.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    root_height: float = 0.8
    debug_vis: bool = False
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class EpisodeLengthCurriculumCfg:
    enable: bool = False
    round_episode_count: int = 1024
    episode_length_ratio: float = 0.98
    required_streak_rounds: int = 3
    speed_increment: float = 0.1
    min_mean_reward: float = 0.0
    # <= 0 means no cap.
    max_forward_speed: float = -1.0
    print_status: bool = True
    stages: list["EpisodeLengthCurriculumStageCfg"] = []


@configclass
class EpisodeLengthCurriculumStageCfg:
    """Per-stage overrides for episode-length speed curriculum."""

    # Number of successful speed updates to stay in this stage.
    # < 0 means infinite (no stage limit).
    max_updates: int = -1
    # Optional command range overrides for this stage.
    lin_vel_x: tuple[float, float] | None = None
    lin_vel_y: tuple[float, float] | None = None
    ang_vel_z: tuple[float, float] | None = None
    # Optional termination override for fall-recovery stages.
    termination_contact_enabled: bool | None = None
    termination_contact_delay_s: float | None = None
    # Optional reset randomization override for joint pose scaling.
    reset_joint_pos_range: tuple[float, float] | None = None
    speed_increment: float | None = None
    min_mean_reward: float | None = None
    episode_length_ratio: float | None = None
    round_episode_count: int | None = None
    required_streak_rounds: int | None = None
    max_forward_speed: float | None = None
    # Optional reward weight overrides for tracking terms.
    track_lin_vel_xy_exp_weight: float | None = None
    track_ang_vel_z_exp_weight: float | None = None
    # Optional reward weight overrides for regularization terms.
    feet_air_time_weight: float | None = None
    energy_weight: float | None = None
    action_rate_l2_weight: float | None = None
    joint_deviation_hip_weight: float | None = None
    joint_deviation_arms_weight: float | None = None
    joint_deviation_legs_weight: float | None = None
    # Optional reward weight schedule for infinite stages.
    track_lin_vel_xy_exp_weight_increment: float | None = None
    track_lin_vel_xy_exp_weight_max: float | None = None
    track_ang_vel_z_exp_weight_increment: float | None = None
    track_ang_vel_z_exp_weight_max: float | None = None


@configclass
class NoiseScalesCfg:
    ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5
    height_scan: float = 0.1


@configclass
class NoiseCfg:
    add_noise: bool = True
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class ActionDelayCfg:
    enable: bool = False
    params: dict = {"max_delay": 5, "min_delay": 0}


@configclass
class DomainRandCfg:
    events: EventCfg = EventCfg()
    action_delay: ActionDelayCfg = ActionDelayCfg()


@configclass
class PhysxCfg:
    gpu_max_rigid_patch_count: int = 10 * 2**15


@configclass
class SimCfg:
    dt: float = 0.005
    decimation: int = 4
    physx: PhysxCfg = PhysxCfg()
