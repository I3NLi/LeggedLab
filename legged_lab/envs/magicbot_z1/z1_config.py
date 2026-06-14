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
SPRINT1_SUBJECT2_MOTION_FILE = (
    "/home/hiyio/whole_body_tracking/motions/magicbot_z1/collected/sprint1_subject2_magicbot_z1.npz"
)
SPRINT_REFERENCE_BODY_NAMES = [
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
SPRINT_AMP_KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
SPRINT_REFERENCE_JOINT_NAMES = [
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_yaw_joint",
]


@configclass
class MagicBotZ1RewardCfg(RewardCfg):
    alive = RewTerm(func=mdp.alive, weight=0.02)
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    forward_speed_progress = RewTerm(
        func=mdp.forward_speed_progress,
        weight=0.0,
        params={"min_command_x": 3.0, "max_ratio": 1.0},
    )
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


@configclass
class MagicBotZ1SprintBridgeRewardCfg(MagicBotZ1RewardCfg):
    sprint_reference_joint_pos = RewTerm(
        func=mdp.reference_joint_pos_exp,
        weight=0.03,
        params={
            "std": 0.65,
            "min_command_speed": 2.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=SPRINT_REFERENCE_JOINT_NAMES),
        },
    )
    sprint_reference_joint_vel = RewTerm(
        func=mdp.reference_joint_vel_exp,
        weight=0.01,
        params={
            "std": 5.0,
            "min_command_speed": 2.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=SPRINT_REFERENCE_JOINT_NAMES),
        },
    )
    sprint_reference_body_relative_pos = RewTerm(
        func=mdp.reference_body_relative_pos_exp,
        weight=0.06,
        params={
            "std": 0.40,
            "min_command_speed": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=SPRINT_REFERENCE_BODY_NAMES),
        },
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


def _apply_magicbot_z1_sprint_reference_motion(env_cfg) -> None:
    env_cfg.reference_motion.enable = True
    env_cfg.reference_motion.motion_file = SPRINT1_SUBJECT2_MOTION_FILE
    env_cfg.reference_motion.anchor_body_name = "torso_link"
    env_cfg.reference_motion.amp_key_body_names = list(SPRINT_AMP_KEY_BODY_NAMES)
    env_cfg.reference_motion.reward_body_names = list(SPRINT_REFERENCE_BODY_NAMES)
    env_cfg.reference_motion.min_command_speed = 2.0
    env_cfg.reference_motion.max_reference_speed = 5.8
    env_cfg.reference_motion.speed_match_tolerance = 0.75
    env_cfg.reference_motion.speed_sample_jitter_frames = 32
    env_cfg.reference_motion.amp_observation_history_length = 2


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
class MagicBotZ1FlatSprintBridgeEnvCfg(MagicBotZ1FlatEnvCfg):
    reward = MagicBotZ1SprintBridgeRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_magicbot_z1_sprint_reference_motion(self)


@configclass
class MagicBotZ1FlatSprintAMPEnvCfg(MagicBotZ1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_magicbot_z1_sprint_reference_motion(self)


@configclass
class MagicBotZ1FlatSprintAMPStage1AEnvCfg(MagicBotZ1FlatSprintAMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.lin_vel_x = (-2.5, 3.0)
        self.reference_motion.max_reference_speed = 3.6


@configclass
class MagicBotZ1FlatSprintAMPStage1BEnvCfg(MagicBotZ1FlatSprintAMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.lin_vel_x = (-2.5, 3.5)
        self.reference_motion.max_reference_speed = 4.0


@configclass
class MagicBotZ1FlatSprintAMPStage1CEnvCfg(MagicBotZ1FlatSprintAMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.lin_vel_x = (-2.5, 3.75)
        self.reference_motion.max_reference_speed = 4.2
        self.reward.track_lin_vel_xy_exp.weight = 1.25


@configclass
class MagicBotZ1FlatSprintAMPStage2AEnvCfg(MagicBotZ1FlatSprintAMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.lin_vel_x = (-2.5, 4.25)
        self.reference_motion.max_reference_speed = 4.8
        self.reward.track_lin_vel_xy_exp.weight = 1.35


@configclass
class MagicBotZ1FlatSprintAMPStage2BEnvCfg(MagicBotZ1FlatSprintAMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.lin_vel_x = (-2.5, 4.75)
        self.reference_motion.max_reference_speed = 5.2
        self.reward.track_lin_vel_xy_exp.weight = 1.4


@configclass
class MagicBotZ1FlatSprintAMPStage2CForwardEnvCfg(MagicBotZ1FlatSprintAMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.rel_standing_envs = 0.05
        self.commands.rel_heading_envs = 0.0
        self.commands.heading_command = False
        self.commands.ranges.lin_vel_x = (0.0, 4.25)
        self.commands.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.ranges.ang_vel_z = (-0.4, 0.4)
        self.commands.ranges.heading = (0.0, 0.0)
        self.reference_motion.max_reference_speed = 4.8
        self.reward.track_lin_vel_xy_exp.weight = 1.35


@configclass
class MagicBotZ1FlatSprintAMPStage2DHighRefEnvCfg(MagicBotZ1FlatSprintAMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.rel_standing_envs = 0.0
        self.commands.rel_heading_envs = 0.0
        self.commands.heading_command = False
        self.commands.ranges.lin_vel_x = (2.5, 4.25)
        self.commands.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.ranges.ang_vel_z = (-0.3, 0.3)
        self.commands.ranges.heading = (0.0, 0.0)
        self.reference_motion.min_command_speed = 3.0
        self.reference_motion.max_reference_speed = 4.8
        self.reference_motion.speed_match_tolerance = 0.5
        self.reward.track_lin_vel_xy_exp.weight = 1.35


@configclass
class MagicBotZ1FlatSprintAMPStage2EProgressEnvCfg(MagicBotZ1FlatSprintAMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.rel_standing_envs = 0.0
        self.commands.rel_heading_envs = 0.0
        self.commands.heading_command = False
        self.commands.ranges.lin_vel_x = (3.0, 4.25)
        self.commands.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.ranges.ang_vel_z = (-0.25, 0.25)
        self.commands.ranges.heading = (0.0, 0.0)
        self.reference_motion.min_command_speed = 3.0
        self.reference_motion.max_reference_speed = 4.8
        self.reference_motion.speed_match_tolerance = 0.5
        self.reward.track_lin_vel_xy_exp.weight = 1.35
        self.reward.track_lin_vel_xy_exp.params["std"] = 0.75
        self.reward.forward_speed_progress.weight = 0.35


@configclass
class MagicBotZ1FlatSprintAMPStage2FPostureEnvCfg(MagicBotZ1FlatSprintAMPStage2EProgressEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.reward.forward_speed_progress.weight = 0.30
        self.reward.ang_vel_xy_l2.weight = -0.08
        self.reward.body_orientation_l2.weight = -2.5
        self.reward.flat_orientation_l2.weight = -1.2
        self.reward.head_shoulder_contact_termination_penalty.weight = -240.0


@configclass
class MagicBotZ1FlatSprintAMPStage2GHoldEnvCfg(MagicBotZ1FlatSprintAMPStage2FPostureEnvCfg):
    pass


@configclass
class MagicBotZ1FlatSprintAMPStage2HSpeedExtendEnvCfg(MagicBotZ1FlatSprintAMPStage2FPostureEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.lin_vel_x = (3.0, 4.5)
        self.reference_motion.max_reference_speed = 5.0
        self.reference_motion.speed_match_tolerance = 0.6
        self.reward.track_lin_vel_xy_exp.params["std"] = 0.8
        self.reward.forward_speed_progress.weight = 0.32


@configclass
class MagicBotZ1FlatSprintAMPStage2IHighTrackEnvCfg(MagicBotZ1FlatSprintAMPStage2FPostureEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.lin_vel_x = (3.5, 4.5)
        self.commands.ranges.lin_vel_y = (-0.08, 0.08)
        self.commands.ranges.ang_vel_z = (-0.18, 0.18)
        self.reference_motion.min_command_speed = 3.5
        self.reference_motion.max_reference_speed = 5.1
        self.reference_motion.speed_match_tolerance = 0.65
        self.reward.track_lin_vel_xy_exp.weight = 1.70
        self.reward.track_lin_vel_xy_exp.params["std"] = 1.0
        self.reward.forward_speed_progress.weight = 0.24
        self.reward.forward_speed_progress.params["min_command_x"] = 3.5
        self.reward.energy.weight = -6.0e-4
        self.reward.action_rate_l2.weight = -7.5e-3


@configclass
class MagicBotZ1FlatSprintAMPStage2JTurnRobustEnvCfg(MagicBotZ1FlatSprintAMPStage2IHighTrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.lin_vel_x = (3.25, 4.5)
        self.commands.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.ranges.ang_vel_z = (-0.45, 0.45)
        self.reference_motion.min_command_speed = 3.25
        self.reference_motion.speed_match_tolerance = 0.75
        self.reward.track_lin_vel_xy_exp.weight = 1.80
        self.reward.track_lin_vel_xy_exp.params["std"] = 1.05
        self.reward.track_ang_vel_z_exp.weight = 1.45
        self.reward.track_ang_vel_z_exp.params["std"] = 0.55
        self.reward.forward_speed_progress.weight = 0.20
        self.reward.forward_speed_progress.params["min_command_x"] = 3.25


@configclass
class MagicBotZ1FlatSprintAMPStage2KMixedRetentionEnvCfg(MagicBotZ1FlatSprintAMPStage2JTurnRobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.lin_vel_x = (3.5, 4.65)
        self.commands.ranges.lin_vel_y = (-0.22, 0.22)
        self.commands.ranges.ang_vel_z = (-0.40, 0.40)
        self.reference_motion.min_command_speed = 3.5
        self.reference_motion.max_reference_speed = 5.3
        self.reference_motion.speed_match_tolerance = 0.70
        self.reward.track_lin_vel_xy_exp.weight = 1.95
        self.reward.track_lin_vel_xy_exp.params["std"] = 0.95
        self.reward.track_ang_vel_z_exp.weight = 1.30
        self.reward.track_ang_vel_z_exp.params["std"] = 0.55
        self.reward.forward_speed_progress.weight = 0.30
        self.reward.forward_speed_progress.params["min_command_x"] = 3.5


@configclass
class MagicBotZ1FlatSprintAMPStage2LStabilityAnchorEnvCfg(MagicBotZ1FlatSprintAMPStage2KMixedRetentionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.ranges.lin_vel_x = (3.4, 4.55)
        self.commands.ranges.lin_vel_y = (-0.18, 0.18)
        self.commands.ranges.ang_vel_z = (-0.32, 0.32)
        self.reference_motion.min_command_speed = 3.4
        self.reference_motion.max_reference_speed = 5.2
        self.reference_motion.speed_match_tolerance = 0.75
        self.reward.track_lin_vel_xy_exp.weight = 1.85
        self.reward.track_lin_vel_xy_exp.params["std"] = 1.05
        self.reward.track_ang_vel_z_exp.weight = 1.20
        self.reward.track_ang_vel_z_exp.params["std"] = 0.60
        self.reward.forward_speed_progress.weight = 0.26
        self.reward.forward_speed_progress.params["min_command_x"] = 3.4


@configclass
class MagicBotZ1FlatAgentCfg(G1FlatAgentCfg):
    experiment_name: str = "magicbot_z1_flat"
    wandb_project: str = "magicbot_z1_flat"


@configclass
class MagicBotZ1FlatSprintBridgeAgentCfg(MagicBotZ1FlatAgentCfg):
    run_name: str = "z1_sprint_bridge_sprint1_subject2"


@configclass
class MagicBotZ1FlatSprintAMPAgentCfg(MagicBotZ1FlatAgentCfg):
    run_name: str = "z1_sprint_amp_sprint1_subject2"

    def __post_init__(self):
        super().__post_init__()
        self.motion_prior.enable = True
        self.motion_prior.num_frames = 2
        self.motion_prior.replay_buffer_size = 200_000
        self.motion_prior.reward_coef = 0.15
        self.motion_prior.reward_min_command_speed = 2.0
        self.motion_prior.discriminator_hidden_dims = [256, 128]
        self.motion_prior.discriminator_learning_rate = 1.0e-4
        self.motion_prior.discriminator_weight_decay = 1.0e-4
        self.motion_prior.grad_penalty_coef = 5.0
        self.motion_prior.use_spectral_norm = True


@configclass
class MagicBotZ1FlatSprintAMPStage1AAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = "z1_sprint_amp_stage1a_cmdx-2p5_3p0_ref2p0_3p6"


@configclass
class MagicBotZ1FlatSprintAMPStage1BAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = "z1_sprint_amp_stage1b_cmdx-2p5_3p5_ref2p0_4p0"


@configclass
class MagicBotZ1FlatSprintAMPStage1CAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = "z1_sprint_amp_stage1c_retention_cmdx-2p5_3p75_ref2p0_4p2"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 5.0e-4
        self.motion_prior.reward_coef = 0.10
        self.save_interval = 25


@configclass
class MagicBotZ1FlatSprintAMPStage2AAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = "z1_sprint_amp_stage2a_cmdx-2p5_4p25_ref2p0_4p8_amp0p08_lr3e-4"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-4
        self.motion_prior.reward_coef = 0.08
        self.save_interval = 25


@configclass
class MagicBotZ1FlatSprintAMPStage2BAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = "z1_sprint_amp_stage2b_cmdx-2p5_4p75_ref2p0_5p2_amp0p08_lr3e-4"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-4
        self.motion_prior.reward_coef = 0.08
        self.save_interval = 25


@configclass
class MagicBotZ1FlatSprintAMPStage2CForwardAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = "z1_sprint_amp_stage2c_forward_cmdx0_4p25_ref2p0_4p8_amp0p08_lr3e-4"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-4
        self.motion_prior.reward_coef = 0.08
        self.save_interval = 25


@configclass
class MagicBotZ1FlatSprintAMPStage2DHighRefAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = "z1_sprint_amp_stage2d_highref_cmdx2p5_4p25_ref3p0_4p8_amp0p08_lr2e-4"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 2.0e-4
        self.motion_prior.reward_coef = 0.08
        self.motion_prior.reward_min_command_speed = 3.0
        self.save_interval = 25


@configclass
class MagicBotZ1FlatSprintAMPStage2EProgressAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = "z1_sprint_amp_stage2e_progress_cmdx3p0_4p25_ref3p0_4p8_prog0p35_amp0p08_lr1e-4"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-4
        self.motion_prior.reward_coef = 0.08
        self.motion_prior.reward_min_command_speed = 3.0
        self.save_interval = 25


@configclass
class MagicBotZ1FlatSprintAMPStage2FPostureAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = "z1_sprint_amp_stage2f_posture_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr7p5e-5"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 7.5e-5
        self.motion_prior.reward_coef = 0.08
        self.motion_prior.reward_min_command_speed = 3.0
        self.save_interval = 25


@configclass
class MagicBotZ1FlatSprintAMPStage2GHoldAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = "z1_sprint_amp_stage2g_hold_cmdx3p0_4p25_ref3p0_4p8_prog0p30_amp0p08_lr5e-5"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 5.0e-5
        self.motion_prior.reward_coef = 0.08
        self.motion_prior.reward_min_command_speed = 3.0
        self.save_interval = 25


@configclass
class MagicBotZ1FlatSprintAMPStage2HSpeedExtendAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = "z1_sprint_amp_stage2h_speedextend_cmdx3p0_4p5_ref3p0_5p0_prog0p32_amp0p08_lr5e-5"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 5.0e-5
        self.motion_prior.reward_coef = 0.08
        self.motion_prior.reward_min_command_speed = 3.0
        self.save_interval = 25


@configclass
class MagicBotZ1FlatSprintAMPStage2IHighTrackAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = (
        "z1_sprint_amp_stage2i_hightrack_cmdx3p5_4p5_ref3p5_5p1_"
        "track1p7_std1p0_prog0p24_energy6e-4_lr4e-5"
    )

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 4.0e-5
        self.motion_prior.reward_coef = 0.08
        self.motion_prior.reward_min_command_speed = 3.5
        self.save_interval = 25


@configclass
class MagicBotZ1FlatSprintAMPStage2JTurnRobustAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = (
        "z1_sprint_amp_stage2j_turnrobust_cmdx3p25_4p5_"
        "cmdy0p25_yaw0p45_trackxy1p8_trackyaw1p45_lr4e-5"
    )

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 4.0e-5
        self.motion_prior.reward_coef = 0.08
        self.motion_prior.reward_min_command_speed = 3.25
        self.save_interval = 25


@configclass
class MagicBotZ1FlatSprintAMPStage2KMixedRetentionAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = (
        "z1_sprint_amp_stage2k_mixedretention_cmdx3p5_4p65_"
        "cmdy0p22_yaw0p40_trackxy1p95_prog0p30_lr3e-5"
    )

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-5
        self.motion_prior.reward_coef = 0.08
        self.motion_prior.reward_min_command_speed = 3.5
        self.save_interval = 25


@configclass
class MagicBotZ1FlatSprintAMPStage2LStabilityAnchorAgentCfg(MagicBotZ1FlatSprintAMPAgentCfg):
    run_name: str = (
        "z1_sprint_amp_stage2l_stabilityanchor_cmdx3p4_4p55_"
        "cmdy0p18_yaw0p32_trackxy1p85_std1p05_prog0p26_lr2e-5"
    )

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 2.0e-5
        self.motion_prior.reward_coef = 0.08
        self.motion_prior.reward_min_command_speed = 3.4
        self.save_interval = 25


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
