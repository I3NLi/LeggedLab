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


# python legged_lab/scripts/play.py --task=g1_flat --num_envs=10 --play_lin_vel_x=3
import argparse
import os

import torch
from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--play_lin_vel_x_min", type=float, default=None, help="Minimum linear velocity command on x-axis.")
parser.add_argument("--play_lin_vel_x", type=float, default=0.6, help="Fixed linear velocity command on x-axis.")
parser.add_argument("--play_lin_vel_y", type=float, default=0.0, help="Fixed linear velocity command on y-axis.")
parser.add_argument("--play_heading", type=float, default=0.0, help="Fixed heading command.")
# Export-only mode is useful for headless batch jobs that only need policy artifacts.
parser.add_argument(
    "--export_only",
    action="store_true",
    help="Export policy to the run's exported/ directory and exit (use with --headless).",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.episode_length_curriculum.enable = False
    env_cfg.commands.debug_vis = True
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    # Keep default backward speed unless a minimum forward speed is provided.
    lin_vel_x_min, _lin_vel_x_max = env_cfg.commands.ranges.lin_vel_x
    if args_cli.play_lin_vel_x_min is not None:
        lin_vel_x_min = args_cli.play_lin_vel_x_min
    env_cfg.commands.ranges.lin_vel_x = (lin_vel_x_min, args_cli.play_lin_vel_x)
    env_cfg.commands.ranges.lin_vel_y = (args_cli.play_lin_vel_y, args_cli.play_lin_vel_y)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (args_cli.play_heading, args_cli.play_heading)
    env_cfg.commands.heading_command = False
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    # env_cfg.scene.terrain_generator = None
    # env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)
    env.command_metrics_enabled = False

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    obs_normalizer = getattr(runner, "obs_normalizer", None)
    export_policy_as_jit(runner.alg.policy, obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )
    if args_cli.export_only:
        print(f"[INFO] Exported policy artifacts to: {export_model_dir}")
        return

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs, _ = obs

    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    play()
    simulation_app.close()
