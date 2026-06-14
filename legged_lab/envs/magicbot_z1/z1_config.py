# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.magicbot import MAGICBOT_Z1_CFG
from legged_lab.envs.base.base_env_config import RewardCfg
from legged_lab.envs.g1.g1_config import G1FlatAgentCfg, G1FlatEnvCfg, G1RoughAgentCfg, G1RoughEnvCfg


FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
NON_FOOT_CONTACT_BODY_NAMES = [r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+"]
HEAD_SHOULDER_CONTACT_BODY_NAMES = [".*head.*", ".*shoulder.*"]


@configclass
class MagicBotZ1RewardCfg(RewardCfg):
    alive = RewTerm(func=mdp.alive, weight=0.02)
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=NON_FOOT_CONTACT_BODY_NAMES), "threshold": 1.0},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FOOT_BODY_NAMES), "threshold": 1.0},
    )
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, weight=-2.0
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    head_shoulder_contact_termination_penalty = RewTerm(
        func=mdp.is_head_shoulder_contact_terminated,
        weight=-180.0,
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FOOT_BODY_NAMES), "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FOOT_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FOOT_BODY_NAMES),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FOOT_BODY_NAMES)},
    )
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


def _apply_magicbot_z1_upstream_commands(env_cfg) -> None:
    env_cfg.episode_length_curriculum.enable = False
    env_cfg.episode_length_curriculum.stages = []
    env_cfg.commands.resampling_time_range = (10.0, 10.0)
    env_cfg.commands.rel_standing_envs = 0.2
    env_cfg.commands.rel_heading_envs = 1.0
    env_cfg.commands.heading_command = True
    env_cfg.commands.heading_control_stiffness = 0.5
    env_cfg.commands.command_slew_rate = (2.0, 1.0, 2.0)
    env_cfg.commands.debug_vis = True
    env_cfg.commands.ranges.lin_vel_x = (-2.5, 5.0)
    env_cfg.commands.ranges.lin_vel_y = (-0.5, 0.5)
    env_cfg.commands.ranges.ang_vel_z = (-1.57, 1.57)


def _apply_magicbot_z1_upstream_reward_weights(env_cfg) -> None:
    env_cfg.reward.alive.weight = 0.02
    env_cfg.reward.track_lin_vel_xy_exp.weight = 1.0
    env_cfg.reward.track_ang_vel_z_exp.weight = 1.0
    env_cfg.reward.lin_vel_z_l2.weight = -1.0
    env_cfg.reward.ang_vel_xy_l2.weight = -0.05
    env_cfg.reward.energy.weight = -1e-3
    env_cfg.reward.dof_acc_l2.weight = -2.5e-7
    env_cfg.reward.action_rate_l2.weight = -0.01
    env_cfg.reward.undesired_contacts.weight = -1.0
    env_cfg.reward.fly.weight = -1.0
    env_cfg.reward.body_orientation_l2.weight = -2.0
    env_cfg.reward.flat_orientation_l2.weight = -1.0
    env_cfg.reward.termination_penalty.weight = -200.0
    env_cfg.reward.head_shoulder_contact_termination_penalty.weight = -180.0
    env_cfg.reward.feet_air_time.weight = 0.15
    env_cfg.reward.feet_slide.weight = -0.25
    env_cfg.reward.feet_force.weight = -3e-3
    env_cfg.reward.feet_too_near.weight = -2.0
    env_cfg.reward.feet_stumble.weight = -2.0
    env_cfg.reward.dof_pos_limits.weight = -2.0
    env_cfg.reward.joint_deviation_hip.weight = -0.15
    env_cfg.reward.joint_deviation_arms.weight = -0.2
    env_cfg.reward.joint_deviation_legs.weight = -0.02


def _apply_magicbot_z1_overrides(env_cfg) -> None:
    env_cfg.scene.robot = MAGICBOT_Z1_CFG
    env_cfg.scene.height_scanner.prim_body_name = "torso_link"

    env_cfg.robot.terminate_contacts_body_names = NON_FOOT_CONTACT_BODY_NAMES
    env_cfg.robot.immediate_terminate_contacts_body_names = HEAD_SHOULDER_CONTACT_BODY_NAMES
    env_cfg.robot.terminate_contacts_delay_s = 1.0
    env_cfg.robot.terminate_contacts_recovery_height = 0.65
    env_cfg.robot.terminate_when_speed_tracking_failed = True
    env_cfg.robot.speed_tracking_command_threshold = 0.5
    env_cfg.robot.speed_tracking_abs_error_threshold = 0.5
    env_cfg.robot.speed_tracking_rel_error_threshold = 0.35
    env_cfg.robot.speed_tracking_grace_s = 2.0
    env_cfg.robot.speed_tracking_duration_s = 2.5
    env_cfg.robot.feet_body_names = FOOT_BODY_NAMES
    # Local base_env has an extra root-height command; 0.0 falls back to the Z1 init height.
    env_cfg.commands.root_height = 0.0

    env_cfg.domain_rand.events.physics_material.params["static_friction_range"] = (0.2, 1.2)
    env_cfg.domain_rand.events.physics_material.params["dynamic_friction_range"] = (0.2, 1.2)
    env_cfg.domain_rand.events.physics_material.params["restitution_range"] = (0.0, 0.0)
    env_cfg.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ".*"
    env_cfg.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (0.9, 1.1)
    env_cfg.domain_rand.events.add_base_mass.params["operation"] = "scale"
    env_cfg.domain_rand.events.add_base_mass.params["recompute_inertia"] = True
    env_cfg.domain_rand.events.add_torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*"),
            "mass_distribution_params": (-1.0, 5.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    env_cfg.domain_rand.events.add_hand_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"]),
            "mass_distribution_params": (-0.5, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    env_cfg.domain_rand.events.randomize_torso_pelvis_com = EventTerm(
        func=mdp.randomize_rigid_body_com_fixed,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "torso_link"]),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)},
        },
    )
    env_cfg.domain_rand.events.randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains_fixed,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    env_cfg.domain_rand.events.randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.15),
            "operation": "abs",
        },
    )
    env_cfg.domain_rand.events.reset_base.params["velocity_range"] = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
    }
    # Use additive reset noise so zero-default joints are randomized too.
    env_cfg.domain_rand.events.reset_robot_joints.func = mdp.reset_joints_by_offset
    env_cfg.domain_rand.events.reset_robot_joints.params["position_range"] = (-0.25, 0.25)
    env_cfg.domain_rand.events.reset_robot_joints.params["velocity_range"] = (-0.5, 0.5)
    env_cfg.domain_rand.events.push_robot.interval_range_s = (10.0, 15.0)
    env_cfg.domain_rand.events.push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
    env_cfg.domain_rand.action_delay.enable = True
    env_cfg.domain_rand.action_delay.params = {"min_delay": 0, "max_delay": 1}

    env_cfg.noise.add_bias = True
    env_cfg.noise.bias_scales.ang_vel = 0.03
    env_cfg.noise.bias_scales.projected_gravity = 0.02
    env_cfg.noise.bias_scales.joint_pos = 0.01
    env_cfg.noise.bias_scales.joint_vel = 0.10


@configclass
class MagicBotZ1FlatEnvCfg(G1FlatEnvCfg):
    reward = MagicBotZ1RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_magicbot_z1_overrides(self)
        _apply_magicbot_z1_upstream_commands(self)
        _apply_magicbot_z1_upstream_reward_weights(self)


@configclass
class MagicBotZ1FlatAgentCfg(G1FlatAgentCfg):
    experiment_name: str = "magicbot_z1_flat"
    wandb_project: str = "magicbot_z1_flat"


@configclass
class MagicBotZ1RoughEnvCfg(G1RoughEnvCfg):
    reward = MagicBotZ1RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_magicbot_z1_overrides(self)


@configclass
class MagicBotZ1RoughAgentCfg(G1RoughAgentCfg):
    experiment_name: str = "magicbot_z1_rough"
    wandb_project: str = "magicbot_z1_rough"
