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


from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.g1.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1GravelAgentCfg,
    G1GravelEnvCfg,
    G1RoughAgentCfg,
    G1RoughEnvCfg,
)
from legged_lab.envs.gr2.gr2_config import (
    GR2FlatAgentCfg,
    GR2FlatEnvCfg,
    GR2RoughAgentCfg,
    GR2RoughEnvCfg,
)
from legged_lab.envs.h1.h1_config import (
    H1FlatAgentCfg,
    H1FlatEnvCfg,
    H1RoughAgentCfg,
    H1RoughEnvCfg,
)
from legged_lab.envs.magicbot_z1.z1_config import (
    MagicBotZ1FlatAgentCfg,
    MagicBotZ1FlatEnvCfg,
    MagicBotZ1FlatSprintAMPAgentCfg,
    MagicBotZ1FlatSprintAMPEnvCfg,
    MagicBotZ1FlatSprintAMPStage1AAgentCfg,
    MagicBotZ1FlatSprintAMPStage1AEnvCfg,
    MagicBotZ1FlatSprintAMPStage1BAgentCfg,
    MagicBotZ1FlatSprintAMPStage1BEnvCfg,
    MagicBotZ1FlatSprintAMPStage1CAgentCfg,
    MagicBotZ1FlatSprintAMPStage1CEnvCfg,
    MagicBotZ1FlatSprintAMPStage2AAgentCfg,
    MagicBotZ1FlatSprintAMPStage2AEnvCfg,
    MagicBotZ1FlatSprintAMPStage2BAgentCfg,
    MagicBotZ1FlatSprintAMPStage2BEnvCfg,
    MagicBotZ1FlatSprintAMPStage2CForwardAgentCfg,
    MagicBotZ1FlatSprintAMPStage2CForwardEnvCfg,
    MagicBotZ1FlatSprintAMPStage2DHighRefAgentCfg,
    MagicBotZ1FlatSprintAMPStage2DHighRefEnvCfg,
    MagicBotZ1FlatSprintAMPStage2EProgressAgentCfg,
    MagicBotZ1FlatSprintAMPStage2EProgressEnvCfg,
    MagicBotZ1FlatSprintAMPStage2FPostureAgentCfg,
    MagicBotZ1FlatSprintAMPStage2FPostureEnvCfg,
    MagicBotZ1FlatSprintAMPStage2GHoldAgentCfg,
    MagicBotZ1FlatSprintAMPStage2GHoldEnvCfg,
    MagicBotZ1FlatSprintAMPStage2HSpeedExtendAgentCfg,
    MagicBotZ1FlatSprintAMPStage2HSpeedExtendEnvCfg,
    MagicBotZ1FlatSprintBridgeAgentCfg,
    MagicBotZ1FlatSprintBridgeEnvCfg,
    MagicBotZ1RoughAgentCfg,
    MagicBotZ1RoughEnvCfg,
)
from legged_lab.utils.task_registry import task_registry

task_registry.register("h1_flat", BaseEnv, H1FlatEnvCfg(), H1FlatAgentCfg())
task_registry.register("h1_rough", BaseEnv, H1RoughEnvCfg(), H1RoughAgentCfg())
task_registry.register("g1_flat", BaseEnv, G1FlatEnvCfg(), G1FlatAgentCfg())
task_registry.register("g1_gravel", BaseEnv, G1GravelEnvCfg(), G1GravelAgentCfg())
task_registry.register("g1_rough", BaseEnv, G1RoughEnvCfg(), G1RoughAgentCfg())
task_registry.register("gr2_flat", BaseEnv, GR2FlatEnvCfg(), GR2FlatAgentCfg())
task_registry.register("gr2_rough", BaseEnv, GR2RoughEnvCfg(), GR2RoughAgentCfg())
task_registry.register("magicbot_z1_flat", BaseEnv, MagicBotZ1FlatEnvCfg(), MagicBotZ1FlatAgentCfg())
task_registry.register(
    "magicbot_z1_flat_sprint_bridge",
    BaseEnv,
    MagicBotZ1FlatSprintBridgeEnvCfg(),
    MagicBotZ1FlatSprintBridgeAgentCfg(),
)
task_registry.register(
    "magicbot_z1_flat_sprint_amp",
    BaseEnv,
    MagicBotZ1FlatSprintAMPEnvCfg(),
    MagicBotZ1FlatSprintAMPAgentCfg(),
)
task_registry.register(
    "magicbot_z1_flat_sprint_amp_stage1a",
    BaseEnv,
    MagicBotZ1FlatSprintAMPStage1AEnvCfg(),
    MagicBotZ1FlatSprintAMPStage1AAgentCfg(),
)
task_registry.register(
    "magicbot_z1_flat_sprint_amp_stage1b",
    BaseEnv,
    MagicBotZ1FlatSprintAMPStage1BEnvCfg(),
    MagicBotZ1FlatSprintAMPStage1BAgentCfg(),
)
task_registry.register(
    "magicbot_z1_flat_sprint_amp_stage1c",
    BaseEnv,
    MagicBotZ1FlatSprintAMPStage1CEnvCfg(),
    MagicBotZ1FlatSprintAMPStage1CAgentCfg(),
)
task_registry.register(
    "magicbot_z1_flat_sprint_amp_stage2a",
    BaseEnv,
    MagicBotZ1FlatSprintAMPStage2AEnvCfg(),
    MagicBotZ1FlatSprintAMPStage2AAgentCfg(),
)
task_registry.register(
    "magicbot_z1_flat_sprint_amp_stage2b",
    BaseEnv,
    MagicBotZ1FlatSprintAMPStage2BEnvCfg(),
    MagicBotZ1FlatSprintAMPStage2BAgentCfg(),
)
task_registry.register(
    "magicbot_z1_flat_sprint_amp_stage2c_forward",
    BaseEnv,
    MagicBotZ1FlatSprintAMPStage2CForwardEnvCfg(),
    MagicBotZ1FlatSprintAMPStage2CForwardAgentCfg(),
)
task_registry.register(
    "magicbot_z1_flat_sprint_amp_stage2d_highref",
    BaseEnv,
    MagicBotZ1FlatSprintAMPStage2DHighRefEnvCfg(),
    MagicBotZ1FlatSprintAMPStage2DHighRefAgentCfg(),
)
task_registry.register(
    "magicbot_z1_flat_sprint_amp_stage2e_progress",
    BaseEnv,
    MagicBotZ1FlatSprintAMPStage2EProgressEnvCfg(),
    MagicBotZ1FlatSprintAMPStage2EProgressAgentCfg(),
)
task_registry.register(
    "magicbot_z1_flat_sprint_amp_stage2f_posture",
    BaseEnv,
    MagicBotZ1FlatSprintAMPStage2FPostureEnvCfg(),
    MagicBotZ1FlatSprintAMPStage2FPostureAgentCfg(),
)
task_registry.register(
    "magicbot_z1_flat_sprint_amp_stage2g_hold",
    BaseEnv,
    MagicBotZ1FlatSprintAMPStage2GHoldEnvCfg(),
    MagicBotZ1FlatSprintAMPStage2GHoldAgentCfg(),
)
task_registry.register(
    "magicbot_z1_flat_sprint_amp_stage2h_speedextend",
    BaseEnv,
    MagicBotZ1FlatSprintAMPStage2HSpeedExtendEnvCfg(),
    MagicBotZ1FlatSprintAMPStage2HSpeedExtendAgentCfg(),
)
task_registry.register("magicbot_z1_rough", BaseEnv, MagicBotZ1RoughEnvCfg(), MagicBotZ1RoughAgentCfg())
