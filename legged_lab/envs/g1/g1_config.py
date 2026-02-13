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
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 1.0})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.8})
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
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        # Termination and feet contact configuration.
        self.robot.terminate_contacts_body_names = [".*torso.*"]
        self.robot.terminate_contacts_delay_s = 1.0
        self.robot.feet_body_names = [".*ankle_roll.*"]
        # Domain randomization target (mass noise on torso).
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]
        # Speed curriculum and observation history.
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.episode_length_curriculum.enable = True
        self.episode_length_curriculum.round_episode_count = 2048
        self.episode_length_curriculum.episode_length_ratio = 1
        self.episode_length_curriculum.required_streak_rounds = 1
        self.episode_length_curriculum.speed_increment = 0.0
        self.episode_length_curriculum.max_forward_speed = 6.0
        self.episode_length_curriculum.min_mean_reward = 25.0
        self.episode_length_curriculum.print_status = True
        # Staged curriculum: each stage overrides selected parameters.
        # Gradually increase forward speed, with special focus on 2~3 m/s.
        self.episode_length_curriculum.stages = [
            # Stage 0: basic walking with small lateral/turn commands.
            EpisodeLengthCurriculumStageCfg(
                max_updates=1,
                # termination_contact_delay_s=1.0,
                lin_vel_x=(-0.6, 0.8),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-0.4, 0.4),
                speed_increment=0.0,
                min_mean_reward=20.0,
            ),
            # Stage 1: fall-recovery practice (reduced command ranges).
            EpisodeLengthCurriculumStageCfg(
                max_updates=1,
                termination_contact_delay_s=1.0,
                lin_vel_x=(-1, 1),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.5, 0.5),
                speed_increment=0.0,
                min_mean_reward=20.0,
            ),
            # Stage 2: moderate speed with limited lateral/turning.
            EpisodeLengthCurriculumStageCfg(
                max_updates=1,
                termination_contact_delay_s=1.0,
                lin_vel_x=(0.8, 1.6),
                lin_vel_y=(-0.6, 0.6),
                ang_vel_z=(-0.6, 0.6),
                speed_increment=0.0,
                min_mean_reward=22.0,
                track_lin_vel_xy_exp_weight=1.5,
                track_ang_vel_z_exp_weight=1.5,
            ),
            # Stage 3: faster band before entering 2~3 m/s focus.
            EpisodeLengthCurriculumStageCfg(
                max_updates=1,
                termination_contact_delay_s=1.0,
                lin_vel_x=(1.6, 2.2),
                lin_vel_y=(-0.6, 0.6),
                ang_vel_z=(-0.6, 0.6),
                speed_increment=0.0,
                min_mean_reward=23.0,
                track_lin_vel_xy_exp_weight=1.5,
                track_ang_vel_z_exp_weight=1.5,
            ),
            # Stage 4: focus on forward speed in a tight band (2.0~2.5 m/s).
            EpisodeLengthCurriculumStageCfg(
                max_updates=10,
                termination_contact_delay_s=1.0,
                lin_vel_x=(2.0, 2.5),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.5, 0.5),
                speed_increment=0.0,
                min_mean_reward=25.0,
                track_lin_vel_xy_exp_weight=2.0,
                track_ang_vel_z_exp_weight=2.0,
            ),
            # Stage 5: focus on forward speed in a tight band (2.5~3.0 m/s).
            EpisodeLengthCurriculumStageCfg(
                max_updates=10,
                termination_contact_delay_s=1.0,
                lin_vel_x=(2.5, 3.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-0.5, 0.5),
                speed_increment=0.0,
                min_mean_reward=25.0,
                track_lin_vel_xy_exp_weight=2.0,
                track_ang_vel_z_exp_weight=2.0,
            ),
              EpisodeLengthCurriculumStageCfg(
                max_updates=2,
                termination_contact_delay_s=1.0,
                lin_vel_x=(-1, 3.0),
                lin_vel_y=(-1, 1),
                ang_vel_z=(-1, 1),
                speed_increment=0.0,
                min_mean_reward=25.0,
                track_lin_vel_xy_exp_weight=2.0,
                track_ang_vel_z_exp_weight=2.0,
            ),
            # Stage 6: 3.0~3.5 m/s.
            EpisodeLengthCurriculumStageCfg(
                max_updates=1,
                termination_contact_delay_s=1.0,
                lin_vel_x=(3.0, 3.5),
                lin_vel_y=(-0.4, 0.4),
                ang_vel_z=(-0.4, 0.4),
                speed_increment=0.0,
                min_mean_reward=25.0,
                track_lin_vel_xy_exp_weight=2.5,
                track_ang_vel_z_exp_weight=2.5,
                energy_weight=-2.5e-4,
                action_rate_l2_weight=-5.0e-3,
                joint_deviation_arms_weight=-0.1,
            ),
            # Stage 7: 3.5~4.0 m/s.
            EpisodeLengthCurriculumStageCfg(
                max_updates=1,
                termination_contact_delay_s=1.0,
                lin_vel_x=(3.5, 4.0),
                lin_vel_y=(-0.4, 0.4),
                ang_vel_z=(-0.4, 0.4),
                speed_increment=0.0,
                min_mean_reward=26.0,
                track_lin_vel_xy_exp_weight=2.7,
                track_ang_vel_z_exp_weight=2.7,
                energy_weight=-2.5e-4,
                action_rate_l2_weight=-5.0e-3,
                joint_deviation_arms_weight=-0.1,
            ),
            EpisodeLengthCurriculumStageCfg(
                max_updates=1,
                termination_contact_delay_s=1.0,
                lin_vel_x=(-1, 4.0),
                lin_vel_y=(-1, 1),
                ang_vel_z=(-1, 1),
                speed_increment=0.0,
                min_mean_reward=26.0,
                track_lin_vel_xy_exp_weight=2.0,
                track_ang_vel_z_exp_weight=2.0,
            ),
            # Stage 8: 4.0~4.5 m/s.
            EpisodeLengthCurriculumStageCfg(
                max_updates=1,
                termination_contact_delay_s=1.0,
                lin_vel_x=(4.0, 4.5),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-0.3, 0.3),
                speed_increment=0.0,
                min_mean_reward=26.0,
                track_lin_vel_xy_exp_weight=2.8,
                track_ang_vel_z_exp_weight=2.8,
                energy_weight=-2.5e-4,
                action_rate_l2_weight=-5.0e-3,
                joint_deviation_arms_weight=-0.1,
            ),
            # Stage 9: 4.5~4.8 m/s.
            EpisodeLengthCurriculumStageCfg(
                max_updates=2,
                termination_contact_delay_s=1.0,
                lin_vel_x=(4.5, 4.8),
                lin_vel_y=(-0.25, 0.25),
                ang_vel_z=(-0.25, 0.25),
                speed_increment=0.0,
                min_mean_reward=27.0,
                track_lin_vel_xy_exp_weight=2.9,
                track_ang_vel_z_exp_weight=2.9,
                energy_weight=-2.0e-4,
                action_rate_l2_weight=-5.0e-3,
                joint_deviation_arms_weight=-0.1,
            ),
            # Stage 10: 4.8~5.1 m/s.
            EpisodeLengthCurriculumStageCfg(
                max_updates=2,
                termination_contact_delay_s=1.0,
                lin_vel_x=(4.8, 5.1),
                lin_vel_y=(-0.2, 0.2),
                ang_vel_z=(-0.2, 0.2),
                speed_increment=0.0,
                min_mean_reward=27.0,
                track_lin_vel_xy_exp_weight=3.0,
                track_ang_vel_z_exp_weight=3.0,
                energy_weight=-2.0e-4,
                action_rate_l2_weight=-5.0e-3,
                joint_deviation_arms_weight=-0.1,
            ),
            # Stage 11: 5.1~5.4 m/s.
            EpisodeLengthCurriculumStageCfg(
                max_updates=2,
                termination_contact_delay_s=1.0,
                lin_vel_x=(5.1, 5.4),
                lin_vel_y=(-0.15, 0.15),
                ang_vel_z=(-0.15, 0.15),
                speed_increment=0.0,
                min_mean_reward=28.0,
                track_lin_vel_xy_exp_weight=3.0,
                track_ang_vel_z_exp_weight=3.0,
                energy_weight=-2.0e-4,
                action_rate_l2_weight=-5.0e-3,
                joint_deviation_arms_weight=-0.1,
            ),
            # Stage 12: 5.4~5.7 m/s.
            EpisodeLengthCurriculumStageCfg(
                max_updates=2,
                termination_contact_delay_s=1.0,
                lin_vel_x=(5.4, 5.7),
                lin_vel_y=(-0.15, 0.15),
                ang_vel_z=(-0.15, 0.15),
                speed_increment=0.0,
                min_mean_reward=28.0,
                track_lin_vel_xy_exp_weight=3.0,
                track_ang_vel_z_exp_weight=3.0,
                energy_weight=-2.0e-4,
                action_rate_l2_weight=-5.0e-3,
                joint_deviation_arms_weight=-0.1,
            ),
            # Stage 13: 5.7~6.0 m/s.
            EpisodeLengthCurriculumStageCfg(
                max_updates=2,
                termination_contact_delay_s=1.0,
                lin_vel_x=(5.7, 6.0),
                lin_vel_y=(-0.1, 0.1),
                ang_vel_z=(-0.1, 0.1),
                speed_increment=0.0,
                min_mean_reward=28.0,
                track_lin_vel_xy_exp_weight=3.0,
                track_ang_vel_z_exp_weight=3.0,
                energy_weight=-2.0e-4,
                action_rate_l2_weight=-5.0e-3,
                joint_deviation_arms_weight=-0.1,
            ),
            # Stage 14: open variable speed with a hard cap at 6 m/s.
            # If rewards stall, slowly increase tracking weights to help convergence.
            EpisodeLengthCurriculumStageCfg(
                max_updates=-1,
                termination_contact_delay_s=1.0,
                lin_vel_x=(-2.0, 6.0),
                lin_vel_y=(-2.0, 2.0),
                ang_vel_z=(-2.0, 2.0),
                speed_increment=0.0,
                max_forward_speed=6.0,
                min_mean_reward=28.0,
                track_lin_vel_xy_exp_weight=3.0,
                track_lin_vel_xy_exp_weight_increment=0.05,
                track_lin_vel_xy_exp_weight_max=3.0,
                track_ang_vel_z_exp_weight=3.0,
                track_ang_vel_z_exp_weight_increment=0.05,
                track_ang_vel_z_exp_weight_max=3.0,
                energy_weight=-2.0e-4,
                action_rate_l2_weight=-5.0e-3,
                joint_deviation_arms_weight=-0.1,
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
class G1RoughEnvCfg(G1FlatEnvCfg):

    def __post_init__(self):
        super().__post_init__()
        # Rough terrain overrides (add height scan + adjust rewards).
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.reward.feet_air_time.weight = 0.25
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        self.reward.lin_vel_z_l2.weight = -0.25


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
