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

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import G1_CFG
from legged_lab.envs.base.base_config import EpisodeLengthCurriculumStageCfg
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class G1RewardCfg(RewardCfg):
    # Command tracking rewards (drive toward commanded linear/angular velocity).
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    track_root_height_exp = RewTerm(func=mdp.track_root_height_exp, weight=0.5, params={"std": 0.08})
    # Stability and smoothness penalties.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # Contact-based penalties/rewards.
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"), "threshold": 1.0},
    )
    shoulder_head_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-50.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*shoulder.*", ".*head.*"]), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, weight=-2.0
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    stuck_command_penalty = RewTerm(func=mdp.stuck_command, weight=-80.0)
    # Gait shaping: encourage alternating single-stance timing.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    # Joint limit and posture regularization.
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*", ".*_shoulder_pitch.*", ".*_elbow.*"]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*waist.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_wrist.*"]
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle.*"])},
    )


@configclass
class G1FlatEnvCfg(BaseEnvCfg):

    reward = G1RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        # Scene and robot wiring.
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        # Termination and feet contact configuration.
        self.robot.terminate_contacts_body_names = [".*torso.*"]
        self.robot.immediate_terminate_contacts_body_names = [".*shoulder.*", ".*head.*"]
        self.robot.terminate_contacts_delay_s = 1.0
        self.robot.feet_body_names = [".*ankle_roll.*"]
        # Domain randomization target (mass noise on torso).
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]
        # Observation history.
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        # Speed-first flat curriculum. Stages keep lateral/yaw commands narrow while
        # pushing forward velocity as aggressively as stability allows.
        self.episode_length_curriculum.enable = True
        self.episode_length_curriculum.round_episode_count = 2048
        self.episode_length_curriculum.episode_length_ratio = 0.95
        self.episode_length_curriculum.required_streak_rounds = 1
        self.episode_length_curriculum.min_mean_reward = 0.0
        self.episode_length_curriculum.max_forward_speed = -1.0
        self.episode_length_curriculum.print_status = True
        self.reward.feet_air_time.weight = 0.25
        self.episode_length_curriculum.stages = [
            EpisodeLengthCurriculumStageCfg(
                max_updates=1,
                round_episode_count=1024,
                episode_length_ratio=0.95,
                min_mean_reward=-20.0,
                lin_vel_x=(-0.5, 1.0),
                lin_vel_y=(-0.25, 0.25),
                ang_vel_z=(-0.3, 0.3),
                feet_air_time_weight=0.25,
            ),
            EpisodeLengthCurriculumStageCfg(
                max_updates=1,
                round_episode_count=1536,
                episode_length_ratio=0.95,
                min_mean_reward=-20.0,
                lin_vel_x=(-0.5, 2.0),
                lin_vel_y=(-0.25, 0.25),
                ang_vel_z=(-0.25, 0.25),
                track_lin_vel_xy_exp_weight=1.5,
                track_ang_vel_z_exp_weight=1.2,
            ),
            EpisodeLengthCurriculumStageCfg(
                max_updates=1,
                round_episode_count=2048,
                episode_length_ratio=0.95,
                min_mean_reward=-10.0,
                reset_joint_pos_range=(0.45, 1.55),
                lin_vel_x=(-0.8, 3.2),
                lin_vel_y=(-0.20, 0.20),
                ang_vel_z=(-0.20, 0.20),
                track_lin_vel_xy_exp_weight=2.0,
                track_ang_vel_z_exp_weight=1.5,
                energy_weight=-6.0e-4,
                action_rate_l2_weight=-7.5e-3,
            ),
            EpisodeLengthCurriculumStageCfg(
                max_updates=2,
                round_episode_count=3072,
                episode_length_ratio=0.95,
                min_mean_reward=0.0,
                reset_joint_pos_range=(0.5, 1.5),
                lin_vel_x=(-1.0, 4.8),
                lin_vel_y=(-0.15, 0.15),
                ang_vel_z=(-0.15, 0.15),
                track_lin_vel_xy_exp_weight=3.0,
                track_ang_vel_z_exp_weight=2.0,
                energy_weight=-3.0e-4,
                action_rate_l2_weight=-5.0e-3,
                joint_deviation_arms_weight=-0.12,
            ),
            EpisodeLengthCurriculumStageCfg(
                max_updates=2,
                round_episode_count=4096,
                episode_length_ratio=0.95,
                min_mean_reward=5.0,
                reset_joint_pos_range=(0.6, 1.4),
                lin_vel_x=(-1.0, 6.5),
                lin_vel_y=(-0.10, 0.10),
                ang_vel_z=(-0.12, 0.12),
                track_lin_vel_xy_exp_weight=4.0,
                track_ang_vel_z_exp_weight=2.5,
                energy_weight=-2.0e-4,
                action_rate_l2_weight=-4.0e-3,
                joint_deviation_arms_weight=-0.08,
                joint_deviation_hip_weight=-0.08,
            ),
            EpisodeLengthCurriculumStageCfg(
                max_updates=-1,
                round_episode_count=4096,
                episode_length_ratio=0.95,
                min_mean_reward=0.0,
                reset_joint_pos_range=(0.6, 1.4),
                lin_vel_x=(-1.0, 8.0),
                lin_vel_y=(-0.20, 0.20),
                ang_vel_z=(-0.20, 0.20),
                track_lin_vel_xy_exp_weight=5.0,
                track_ang_vel_z_exp_weight=3.0,
                energy_weight=-1.5e-4,
                action_rate_l2_weight=-3.0e-3,
                joint_deviation_arms_weight=-0.06,
                joint_deviation_hip_weight=-0.05,
            ),
        ]

        # # Emphasize tracking terms for flat ground.
        # self.reward.track_lin_vel_xy_exp.weight = 1.5
        # self.reward.track_ang_vel_z_exp.weight = 1.5

@configclass
class G1FlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_flat"
    wandb_project: str = "g1_flat"


@configclass
class G1GravelEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Small generated gravel terrain for bounded-terrain robustness training.
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.episode_length_curriculum.enable = False
        self.episode_length_curriculum.stages = []


@configclass
class G1GravelAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_gravel"
    wandb_project: str = "g1_gravel"


@configclass
class G1RoughEnvCfg(G1FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        # Rough terrain overrides (add height scan + adjust rewards).
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.episode_length_curriculum.enable = False
        self.episode_length_curriculum.stages = []
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.reward.feet_air_time.weight = 0.25
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        self.reward.lin_vel_z_l2.weight = 0.0


@configclass
class G1RoughAgentCfg(BaseAgentCfg):
    experiment_name: str = "g1_rough"
    wandb_project: str = "g1_rough"

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"
