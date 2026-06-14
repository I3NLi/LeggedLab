import argparse
import os
import statistics

import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a trained policy with fixed velocity commands.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--load_run", type=str, required=True, help="Run folder to load from.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint filename to load.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--duration", type=float, default=8.0, help="Measured duration per speed in seconds.")
parser.add_argument("--warmup", type=float, default=3.0, help="Warmup duration before measuring.")
parser.add_argument("--speeds", nargs="+", type=float, required=True, help="Fixed x velocity commands to evaluate.")
parser.add_argument("--lin_vel_y", type=float, default=0.0, help="Fixed y velocity command.")
parser.add_argument("--ang_vel_z", type=float, default=0.0, help="Fixed yaw-rate command.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.utils import task_registry  # noqa: E402


def _configure_env(env_cfg, num_envs: int):
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.episode_length_curriculum.enable = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_y = (args_cli.lin_vel_y, args_cli.lin_vel_y)
    env_cfg.commands.ranges.ang_vel_z = (args_cli.ang_vel_z, args_cli.ang_vel_z)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.max_episode_length_s = 20.0
    env_cfg.scene.seed = 42

    if hasattr(env_cfg.scene, "height_scanner"):
        env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)


def _force_command(env, vx: float, vy: float, wz: float):
    target = torch.zeros((env.num_envs, 3), device=env.device)
    target[:, 0] = vx
    target[:, 1] = vy
    target[:, 2] = wz
    env.command_generator.vel_command_b[:] = target
    env.command_generator.is_standing_env[:] = False
    env.command_generator.is_heading_env[:] = False
    if hasattr(env.command_generator, "time_left"):
        env.command_generator.time_left[:] = 1e6
    if getattr(env, "_command_slew_enabled", False):
        env._command_buf[:] = target


def _root_command_state(env):
    root_quat_w = env._tensor(env.robot.data.root_quat_w)
    root_lin_vel_w = env._tensor(env.robot.data.root_lin_vel_w)
    root_lin_vel_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(root_quat_w), root_lin_vel_w[:, :3])
    root_ang_vel_w = env._tensor(env.robot.data.root_ang_vel_w)
    return root_lin_vel_yaw[:, 0], root_lin_vel_yaw[:, 1], root_ang_vel_w[:, 2]


def _build_env_and_policy():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    _configure_env(env_cfg, args_cli.num_envs)
    agent_cfg.load_run = args_cli.load_run
    agent_cfg.load_checkpoint = args_cli.checkpoint

    print("[INFO] Building eval env", flush=True)
    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)
    env.command_metrics_enabled = False
    if hasattr(env.robot, "root_view"):
        env.robot.data._root_view = env.robot.root_view
    _install_reset_reason_counter(env)

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=os.path.dirname(resume_path), device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    print("[INFO] Policy loaded", flush=True)
    return env, policy


def _zero_reset_reason_counts():
    return {
        "timeout_resets": 0,
        "head_shoulder_resets": 0,
        "body_contact_resets": 0,
        "speed_tracking_resets": 0,
        "other_resets": 0,
    }


def _install_reset_reason_counter(env):
    if not hasattr(env, "_log_reset_reasons") or getattr(env, "_eval_reset_reason_counter_installed", False):
        return

    original_log_reset_reasons = env._log_reset_reasons
    env._eval_reset_reason_totals = _zero_reset_reason_counts()

    def _counting_log_reset_reasons(env_ids):
        original_log_reset_reasons(env_ids)
        totals = getattr(env, "_eval_reset_reason_totals", None)
        if totals is None:
            return
        log = env.extras.get("log", {})
        totals["timeout_resets"] += int(round(float(log.get("Reset/timeout_count", 0.0))))
        totals["head_shoulder_resets"] += int(round(float(log.get("Reset/head_shoulder_contact_count", 0.0))))
        totals["body_contact_resets"] += int(round(float(log.get("Reset/body_contact_count", 0.0))))
        totals["speed_tracking_resets"] += int(round(float(log.get("Reset/speed_tracking_failure_count", 0.0))))
        totals["other_resets"] += int(round(float(log.get("Reset/other_count", 0.0))))

    env._log_reset_reasons = _counting_log_reset_reasons
    env._eval_reset_reason_counter_installed = True


def _reset_reason_counts(env):
    totals = getattr(env, "_eval_reset_reason_totals", None)
    if totals is None:
        return {
            "timeout_resets": 0,
            "head_shoulder_resets": 0,
            "body_contact_resets": 0,
            "speed_tracking_resets": 0,
            "other_resets": 0,
        }
    return dict(totals)


def _run_one_speed(env, policy, vx: float):
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset(env_ids)
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs, _ = obs
    if hasattr(env, "_eval_reset_reason_totals"):
        env._eval_reset_reason_totals = _zero_reset_reason_counts()

    warmup_steps = int(args_cli.warmup / env.step_dt)
    sample_steps = int(args_cli.duration / env.step_dt)
    vx_samples = []
    vy_samples = []
    wz_samples = []
    xy_err_samples = []
    resets = 0

    print(
        f"[INFO] Evaluating target_vx={vx:.3f} target_vy={args_cli.lin_vel_y:.3f} "
        f"target_wz={args_cli.ang_vel_z:.3f}",
        flush=True,
    )
    with torch.inference_mode():
        for step in range(warmup_steps + sample_steps):
            _force_command(env, vx, args_cli.lin_vel_y, args_cli.ang_vel_z)
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            if isinstance(obs, tuple):
                obs, _ = obs
            if dones is not None:
                resets += int(dones.sum().item())
            _force_command(env, vx, args_cli.lin_vel_y, args_cli.ang_vel_z)
            if step >= warmup_steps:
                root_vx, root_vy, root_wz = _root_command_state(env)
                vx_tensor = root_vx.detach().float()
                vy_tensor = root_vy.detach().float()
                wz_tensor = root_wz.detach().float()
                xy_err = torch.sqrt((vx_tensor - vx) ** 2 + (vy_tensor - args_cli.lin_vel_y) ** 2)
                vx_samples.extend(vx_tensor.cpu().tolist())
                vy_samples.extend(vy_tensor.cpu().tolist())
                wz_samples.extend(wz_tensor.cpu().tolist())
                xy_err_samples.extend(xy_err.cpu().tolist())

    print(f"[INFO] Finished target_vx={vx:.3f}", flush=True)
    reset_reasons = _reset_reason_counts(env)

    if not vx_samples:
        return {
            "target": vx,
            "target_vy": args_cli.lin_vel_y,
            "target_wz": args_cli.ang_vel_z,
            "mean_vx": 0.0,
            "mean_vy": 0.0,
            "mean_wz": 0.0,
            "mean_abs_vx": 0.0,
            "abs_err": 0.0,
            "vy_abs_err": 0.0,
            "wz_abs_err": 0.0,
            "xy_abs_err": 0.0,
            "p50_vx": 0.0,
            "p90_abs_vx": 0.0,
            "p90_xy_err": 0.0,
            "p90_wz_abs_err": 0.0,
            "resets": resets,
            **reset_reasons,
        }

    abs_samples = sorted(abs(value) for value in vx_samples)
    sorted_xy_err = sorted(xy_err_samples)
    sorted_wz_abs_err = sorted(abs(value - args_cli.ang_vel_z) for value in wz_samples)
    p90_idx = int(0.9 * (len(abs_samples) - 1))
    p90_err_idx = int(0.9 * (len(sorted_xy_err) - 1))
    return {
        "target": vx,
        "target_vy": args_cli.lin_vel_y,
        "target_wz": args_cli.ang_vel_z,
        "mean_vx": statistics.fmean(vx_samples),
        "mean_vy": statistics.fmean(vy_samples),
        "mean_wz": statistics.fmean(wz_samples),
        "mean_abs_vx": statistics.fmean(abs_samples),
        "abs_err": statistics.fmean(abs(value - vx) for value in vx_samples),
        "vy_abs_err": statistics.fmean(abs(value - args_cli.lin_vel_y) for value in vy_samples),
        "wz_abs_err": statistics.fmean(abs(value - args_cli.ang_vel_z) for value in wz_samples),
        "xy_abs_err": statistics.fmean(xy_err_samples),
        "p50_vx": statistics.median(vx_samples),
        "p90_abs_vx": abs_samples[p90_idx],
        "p90_xy_err": sorted_xy_err[p90_err_idx],
        "p90_wz_abs_err": sorted_wz_abs_err[p90_err_idx],
        "resets": resets,
        **reset_reasons,
    }


def main():
    print("[INFO] Fixed-speed eval start", flush=True)
    env, policy = _build_env_and_policy()
    try:
        for speed in args_cli.speeds:
            result = _run_one_speed(env, policy, speed)
            print(
                "RESULT "
                + " ".join(
                    f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in result.items()
                ),
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
