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


"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

# Use T1 assets directly from HoloMotion's vendored GMR repository.
HOLOMOTION_GMR_ROOT = Path(os.environ.get("HOLOMOTION_GMR_ROOT", "/home/hiyio/HoloMotion/thirdparties/GMR"))
T1_URDF_PATH = HOLOMOTION_GMR_ROOT / "assets" / "booster_t1" / "T1_serial.urdf"

H1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/unitree/h1/h1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_pitch_joint": -0.28,
            ".*_knee_joint": 0.79,
            ".*_ankle_joint": -0.52,
            ".*_shoulder_pitch_joint": 0.20,
            ".*_elbow_joint": 0.32,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*torso_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 300.0,
                ".*torso_joint": 200.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 23.0,
                ".*_hip_roll_joint": 23.0,
                ".*_hip_pitch_joint": 23.0,
                ".*_knee_joint": 14.0,
                ".*torso_joint": 23.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 300.0,
                ".*torso_joint": 300.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 6.0,
                ".*torso_joint": 6.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_joint"],
            effort_limit_sim={".*_ankle_joint": 40.0},
            velocity_limit_sim={".*_ankle_joint": 9.0},
            stiffness={".*_ankle_joint": 40.0},
            damping={".*_ankle_joint": 2.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 18.0,
                ".*_elbow_joint": 18.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 9.0,
                ".*_shoulder_roll_joint": 9.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 20.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 50.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
            },
        ),
    },
)


G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/unitree/g1/g1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                ".*waist_yaw_joint": 88.0,
                ".*waist_roll_joint": 35.0,
                ".*waist_pitch_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                ".*waist_yaw_joint": 32.0,
                ".*waist_roll_joint": 30.0,
                ".*waist_pitch_joint": 30.0,
            },
            stiffness={
                "left_hip_pitch_joint": 40.179,
                "left_hip_roll_joint": 40.179,
                "left_hip_yaw_joint": 40.179,
                "left_knee_joint": 99.098,
                "right_hip_pitch_joint": 40.179,
                "right_hip_roll_joint": 40.179,
                "right_hip_yaw_joint": 28.501,
                "right_knee_joint": 99.098,
                "waist_yaw_joint": 14.251,
                "waist_roll_joint": 28.501,
                "waist_pitch_joint": 28.501,
            },
            damping={
                "left_hip_pitch_joint": 2.558,
                "left_hip_roll_joint": 2.558,
                "left_hip_yaw_joint": 2.558,
                "left_knee_joint": 6.309,
                "right_hip_pitch_joint": 2.558,
                "right_hip_roll_joint": 2.558,
                "right_hip_yaw_joint": 1.814,
                "right_knee_joint": 6.309,
                "waist_yaw_joint": 0.907,
                "waist_roll_joint": 1.814,
                "waist_pitch_joint": 1.814,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 35.0,
                ".*_ankle_roll_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint": 30.0,
            },
            stiffness={
                "left_ankle_pitch_joint": 99.098,
                "left_ankle_roll_joint": 28.501,
                "right_ankle_pitch_joint": 99.098,
                "right_ankle_roll_joint": 14.251,
            },
            damping={
                "left_ankle_pitch_joint": 6.309,
                "left_ankle_roll_joint": 1.814,
                "right_ankle_pitch_joint": 6.309,
                "right_ankle_roll_joint": 0.907,
            },
            armature=0.01,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
            },
            stiffness={
                "left_shoulder_pitch_joint": 14.251,
                "left_shoulder_roll_joint": 14.251,
                "right_shoulder_pitch_joint": 14.251,
                "right_shoulder_roll_joint": 14.251,
            },
            damping={
                "left_shoulder_pitch_joint": 0.907,
                "left_shoulder_roll_joint": 0.907,
                "right_shoulder_pitch_joint": 0.907,
                "right_shoulder_roll_joint": 0.907,
            },
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
            },
            stiffness={
                "left_shoulder_yaw_joint": 28.501,
                "left_elbow_joint": 28.501,
                "right_shoulder_yaw_joint": 14.251,
                "right_elbow_joint": 14.251,
            },
            damping={
                "left_shoulder_yaw_joint": 1.814,
                "left_elbow_joint": 1.814,
                "right_shoulder_yaw_joint": 0.907,
                "right_elbow_joint": 0.907,
            },
            armature=0.01,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_.*",
            ],
            effort_limit_sim={
                ".*_wrist_yaw_joint": 5.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_wrist_yaw_joint": 22.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
            },
            stiffness={
                "left_wrist_roll_joint": 14.251,
                "left_wrist_pitch_joint": 14.251,
                "left_wrist_yaw_joint": 14.251,
                "right_wrist_roll_joint": 16.778,
                "right_wrist_pitch_joint": 16.778,
                "right_wrist_yaw_joint": 16.778,
            },
            damping={
                "left_wrist_roll_joint": 0.907,
                "left_wrist_pitch_joint": 0.907,
                "left_wrist_yaw_joint": 0.907,
                "right_wrist_roll_joint": 1.068,
                "right_wrist_pitch_joint": 1.068,
                "right_wrist_yaw_joint": 1.068,
            },
            armature=0.01,
        ),
    },
)


T1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(T1_URDF_PATH),
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.90),
        joint_pos={
            "Left_Hip_Pitch": -0.20,
            "Right_Hip_Pitch": -0.20,
            "Left_Knee_Pitch": 0.42,
            "Right_Knee_Pitch": 0.42,
            "Left_Ankle_Pitch": -0.22,
            "Right_Ankle_Pitch": -0.22,
            "Left_Shoulder_Pitch": 0.25,
            "Right_Shoulder_Pitch": 0.25,
            "Left_Shoulder_Roll": 0.10,
            "Right_Shoulder_Roll": -0.10,
            "Left_Elbow_Pitch": 0.50,
            "Right_Elbow_Pitch": 0.50,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            effort_limit_sim=7.0,
            velocity_limit_sim=12.56,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "Left_Shoulder_Pitch",
                "Left_Shoulder_Roll",
                "Left_Elbow_Pitch",
                "Left_Elbow_Yaw",
                "Right_Shoulder_Pitch",
                "Right_Shoulder_Roll",
                "Right_Elbow_Pitch",
                "Right_Elbow_Yaw",
            ],
            effort_limit_sim=18.0,
            velocity_limit_sim=18.84,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["Waist"],
            effort_limit_sim=30.0,
            velocity_limit_sim=18.84,
            stiffness=35.0,
            damping=1.5,
            armature=0.01,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "Left_Hip_Pitch",
                "Left_Hip_Roll",
                "Left_Hip_Yaw",
                "Left_Knee_Pitch",
                "Right_Hip_Pitch",
                "Right_Hip_Roll",
                "Right_Hip_Yaw",
                "Right_Knee_Pitch",
            ],
            effort_limit_sim={
                "Left_Hip_Pitch": 45.0,
                "Left_Hip_Roll": 30.0,
                "Left_Hip_Yaw": 30.0,
                "Left_Knee_Pitch": 60.0,
                "Right_Hip_Pitch": 45.0,
                "Right_Hip_Roll": 30.0,
                "Right_Hip_Yaw": 30.0,
                "Right_Knee_Pitch": 60.0,
            },
            velocity_limit_sim=18.84,
            stiffness=80.0,
            damping=4.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "Left_Ankle_Pitch",
                "Left_Ankle_Roll",
                "Right_Ankle_Pitch",
                "Right_Ankle_Roll",
            ],
            effort_limit_sim={
                "Left_Ankle_Pitch": 20.0,
                "Left_Ankle_Roll": 15.0,
                "Right_Ankle_Pitch": 20.0,
                "Right_Ankle_Roll": 15.0,
            },
            velocity_limit_sim=18.84,
            stiffness=60.0,
            damping=3.0,
            armature=0.01,
        ),
    },
)
