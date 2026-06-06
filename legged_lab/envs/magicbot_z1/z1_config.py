# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

from legged_lab.assets.magicbot import MAGICBOT_Z1_CFG
from legged_lab.envs.g1.g1_config import G1FlatAgentCfg, G1FlatEnvCfg, G1RoughAgentCfg, G1RoughEnvCfg


def _apply_magicbot_z1_overrides(env_cfg) -> None:
    env_cfg.scene.robot = MAGICBOT_Z1_CFG
    env_cfg.scene.height_scanner.prim_body_name = "torso_link"

    env_cfg.robot.terminate_contacts_body_names = [".*torso.*", "pelvis", ".*shoulder.*", ".*head.*"]
    env_cfg.robot.terminate_contacts_delay_s = 0.0
    env_cfg.robot.feet_body_names = [".*ankle_roll.*"]

    env_cfg.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]

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
