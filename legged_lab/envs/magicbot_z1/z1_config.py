# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.assets.magicbot import MAGICBOT_Z1_CFG
from legged_lab.envs.base.base_config import EpisodeLengthCurriculumStageCfg
from legged_lab.envs.g1.g1_config import G1FlatAgentCfg, G1FlatEnvCfg, G1RoughAgentCfg, G1RoughEnvCfg


def _apply_magicbot_z1_flat_recovery_curriculum(env_cfg) -> None:
    env_cfg.episode_length_curriculum.enable = True
    env_cfg.episode_length_curriculum.round_episode_count = 2048
    env_cfg.episode_length_curriculum.episode_length_ratio = 0.95
    env_cfg.episode_length_curriculum.required_streak_rounds = 1
    env_cfg.episode_length_curriculum.min_mean_reward = -40.0
    env_cfg.episode_length_curriculum.max_forward_speed = -1.0
    env_cfg.episode_length_curriculum.print_status = True
    env_cfg.episode_length_curriculum.stages = [
        # Fall-recovery bootcamp: keep episodes alive after body contact so
        # the policy can learn to stand back up instead of only learning to reset.
        EpisodeLengthCurriculumStageCfg(
            max_updates=1,
            round_episode_count=1024,
            episode_length_ratio=0.75,
            min_mean_reward=-120.0,
            termination_contact_enabled=False,
            reset_joint_pos_range=(0.20, 1.80),
            lin_vel_x=(-0.3, 0.6),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-0.4, 0.4),
            track_lin_vel_xy_exp_weight=0.8,
            track_ang_vel_z_exp_weight=0.8,
            energy_weight=-8.0e-4,
            action_rate_l2_weight=-8.0e-3,
            joint_deviation_arms_weight=-0.20,
        ),
        # Immediate-termination walking gate.
        EpisodeLengthCurriculumStageCfg(
            max_updates=1,
            round_episode_count=1536,
            episode_length_ratio=0.95,
            min_mean_reward=-60.0,
            termination_contact_enabled=True,
            termination_contact_delay_s=0.0,
            reset_joint_pos_range=(0.35, 1.65),
            lin_vel_x=(-0.5, 1.2),
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-0.3, 0.3),
            track_lin_vel_xy_exp_weight=1.2,
            track_ang_vel_z_exp_weight=1.0,
        ),
        # Impact-recovery stage with stronger commands.
        EpisodeLengthCurriculumStageCfg(
            max_updates=1,
            round_episode_count=2048,
            episode_length_ratio=0.75,
            min_mean_reward=-100.0,
            termination_contact_enabled=True,
            termination_contact_delay_s=1.0,
            reset_joint_pos_range=(0.15, 1.90),
            lin_vel_x=(-0.8, 2.0),
            lin_vel_y=(-0.35, 0.35),
            ang_vel_z=(-0.4, 0.4),
            track_lin_vel_xy_exp_weight=1.5,
            track_ang_vel_z_exp_weight=1.2,
            energy_weight=-6.0e-4,
            action_rate_l2_weight=-6.0e-3,
            joint_deviation_arms_weight=-0.15,
        ),
        EpisodeLengthCurriculumStageCfg(
            max_updates=1,
            round_episode_count=2048,
            episode_length_ratio=0.95,
            min_mean_reward=-30.0,
            termination_contact_enabled=True,
            termination_contact_delay_s=1.0,
            reset_joint_pos_range=(0.45, 1.55),
            lin_vel_x=(-1.0, 3.0),
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-0.25, 0.25),
            track_lin_vel_xy_exp_weight=2.0,
            track_ang_vel_z_exp_weight=1.5,
            energy_weight=-5.0e-4,
            action_rate_l2_weight=-5.0e-3,
        ),
        # Last recovery pass before high-speed training.
        EpisodeLengthCurriculumStageCfg(
            max_updates=1,
            round_episode_count=3072,
            episode_length_ratio=0.80,
            min_mean_reward=-80.0,
            termination_contact_enabled=True,
            termination_contact_delay_s=1.0,
            reset_joint_pos_range=(0.25, 1.75),
            lin_vel_x=(-1.0, 4.0),
            lin_vel_y=(-0.25, 0.25),
            ang_vel_z=(-0.25, 0.25),
            track_lin_vel_xy_exp_weight=2.5,
            track_ang_vel_z_exp_weight=1.8,
            energy_weight=-4.0e-4,
            action_rate_l2_weight=-4.5e-3,
            joint_deviation_arms_weight=-0.12,
        ),
        EpisodeLengthCurriculumStageCfg(
            max_updates=2,
            round_episode_count=4096,
            episode_length_ratio=0.95,
            min_mean_reward=-20.0,
            termination_contact_enabled=True,
            termination_contact_delay_s=1.0,
            reset_joint_pos_range=(0.50, 1.50),
            lin_vel_x=(-1.0, 5.5),
            lin_vel_y=(-0.20, 0.20),
            ang_vel_z=(-0.20, 0.20),
            track_lin_vel_xy_exp_weight=3.5,
            track_ang_vel_z_exp_weight=2.2,
            energy_weight=-2.5e-4,
            action_rate_l2_weight=-4.0e-3,
            joint_deviation_arms_weight=-0.10,
        ),
        EpisodeLengthCurriculumStageCfg(
            max_updates=-1,
            round_episode_count=4096,
            episode_length_ratio=0.95,
            min_mean_reward=-20.0,
            termination_contact_enabled=True,
            termination_contact_delay_s=1.0,
            reset_joint_pos_range=(0.60, 1.40),
            lin_vel_x=(-1.0, 8.0),
            lin_vel_y=(-0.20, 0.20),
            ang_vel_z=(-0.20, 0.20),
            track_lin_vel_xy_exp_weight=5.0,
            track_ang_vel_z_exp_weight=3.0,
            energy_weight=-1.5e-4,
            action_rate_l2_weight=-3.0e-3,
            joint_deviation_arms_weight=-0.06,
            joint_deviation_hip_weight=-0.05,
            joint_deviation_legs_weight=-0.01,
        ),
    ]


def _apply_magicbot_z1_overrides(env_cfg) -> None:
    env_cfg.scene.robot = MAGICBOT_Z1_CFG
    env_cfg.scene.height_scanner.prim_body_name = "torso_link"

    env_cfg.robot.terminate_contacts_body_names = [".*torso.*", "pelvis"]
    env_cfg.robot.immediate_terminate_contacts_body_names = [".*shoulder.*", ".*head.*"]
    env_cfg.robot.terminate_contacts_delay_s = 0.0
    env_cfg.robot.terminate_when_stuck = True
    env_cfg.robot.stuck_command_threshold = 0.25
    env_cfg.robot.stuck_speed_threshold = 0.08
    env_cfg.robot.stuck_grace_s = 1.0
    env_cfg.robot.stuck_duration_s = 0.6
    env_cfg.robot.feet_body_names = [".*ankle_roll.*"]

    env_cfg.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
    env_cfg.domain_rand.events.reset_base.params["velocity_range"] = {
        "x": (-1.0, 1.0),
        "y": (-1.0, 1.0),
        "z": (-0.7, 0.7),
        "roll": (-1.5, 1.5),
        "pitch": (-1.5, 1.5),
        "yaw": (-1.0, 1.0),
    }
    env_cfg.domain_rand.events.push_robot.interval_range_s = (3.0, 6.0)
    env_cfg.domain_rand.events.push_robot.params["velocity_range"] = {"x": (-2.5, 2.5), "y": (-2.0, 2.0)}

    env_cfg.reward.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_sensor",
        body_names=[
            r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
            r"(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
        ],
    )
    env_cfg.reward.fly.params["sensor_cfg"] = SceneEntityCfg(
        "contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    )
    env_cfg.reward.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
        "contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    )
    env_cfg.reward.feet_slide.params["sensor_cfg"] = SceneEntityCfg(
        "contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    )
    env_cfg.reward.feet_slide.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    )
    env_cfg.reward.feet_force.params["sensor_cfg"] = SceneEntityCfg(
        "contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    )
    env_cfg.reward.feet_too_near.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    )
    env_cfg.reward.feet_stumble.params["sensor_cfg"] = SceneEntityCfg(
        "contact_sensor", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
    )
    env_cfg.reward.body_orientation_l2.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["torso_link"])


@configclass
class MagicBotZ1FlatEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_magicbot_z1_overrides(self)
        _apply_magicbot_z1_flat_recovery_curriculum(self)


@configclass
class MagicBotZ1FlatAgentCfg(G1FlatAgentCfg):
    experiment_name: str = "magicbot_z1_flat"
    wandb_project: str = "magicbot_z1_flat"


@configclass
class MagicBotZ1RoughEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_magicbot_z1_overrides(self)


@configclass
class MagicBotZ1RoughAgentCfg(G1RoughAgentCfg):
    experiment_name: str = "magicbot_z1_rough"
    wandb_project: str = "magicbot_z1_rough"
