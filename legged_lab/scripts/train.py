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

# python legged_lab/scripts/train.py --task=g1_flat --num_envs=4096 --resume=True --load_run=latest --load_checkpoint=latest
import argparse

from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner

from legged_lab.amp import AMPOnPolicyRunner
from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--deploy_yaml_root",
    type=str,
    default=None,
    help="Deploy repo root to receive generated MagicBot Z1 LocoMode YAML files.",
)
parser.add_argument(
    "--skip_deploy_yaml",
    action="store_true",
    help="Skip generating MagicBot Z1 deploy YAML files during training startup.",
)
parser.add_argument(
    "--reset_optimizer",
    action="store_true",
    help="When resuming, load policy weights but start with the current task optimizer and learning-rate config.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from legged_lab.utils.magicbot_deploy_yaml import export_magicbot_z1_deploy_yamls

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def train():
    runner: OnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.device is not None:
        env_cfg.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.scene.seed = agent_cfg.seed

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    env = env_class(env_cfg, args_cli.headless)
    # Log key config switches for experiment tracking.
    activation_name = str(getattr(agent_cfg.policy, "activation", "unknown"))
    env._static_log_info["Config/policy_activation_silu"] = float(activation_name.lower() == "silu")
    print(f"[INFO] Policy activation: {activation_name}")
    print(f"[INFO] Termination delay (s): {env_cfg.robot.terminate_contacts_delay_s}")

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    motion_prior_cfg = getattr(agent_cfg, "motion_prior", None)
    motion_prior_enabled = bool(getattr(motion_prior_cfg, "enable", False))
    runner_cls = AMPOnPolicyRunner if motion_prior_enabled else OnPolicyRunner
    print(f"[INFO] Motion prior AMP runner: {motion_prior_enabled}")
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        print(f"[INFO]: Load optimizer state: {not args_cli.reset_optimizer}")
        # load previously trained model
        runner.load(resume_path, load_optimizer=not args_cli.reset_optimizer)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if not args_cli.skip_deploy_yaml:
        deploy_yaml_paths = export_magicbot_z1_deploy_yamls(
            env_class_name,
            env_cfg,
            log_dir=log_dir,
            deploy_root=args_cli.deploy_yaml_root,
        )
        if deploy_yaml_paths:
            print("[INFO] Generated MagicBot Z1 deploy YAML files:")
            for path in deploy_yaml_paths:
                print(f"  {path}")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train()
    simulation_app.close()
