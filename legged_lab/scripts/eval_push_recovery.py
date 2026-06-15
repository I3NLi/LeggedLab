import argparse
import os
import statistics

import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate low-speed recovery under commanded velocity pushes.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--load_run", type=str, required=True, help="Run folder to load from.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint filename to load.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--duration", type=float, default=8.0, help="Measured duration per command in seconds.")
parser.add_argument("--warmup", type=float, default=2.0, help="Warmup duration before pushing.")
parser.add_argument("--speeds", nargs="+", type=float, required=True, help="Fixed x velocity commands to evaluate.")
parser.add_argument("--lin_vel_y", type=float, default=0.0, help="Fixed y velocity command.")
parser.add_argument("--ang_vel_z", type=float, default=0.0, help="Fixed yaw-rate command.")
parser.add_argument(
    "--profiles",
    nargs="+",
    default=["forward", "backward", "left", "right", "yaw_left", "yaw_right"],
    choices=["forward", "backward", "left", "right", "diag_left", "diag_right", "yaw_left", "yaw_right"],
    help="Push profiles to cycle through.",
)
parser.add_argument("--push_interval", type=float, default=1.5, help="Seconds between pushes during measurement.")
parser.add_argument("--recovery_window", type=float, default=1.0, help="Seconds allowed for recovery after each push.")
parser.add_argument("--push_linear_magnitude", type=float, default=1.5, help="Body-frame linear velocity impulse.")
parser.add_argument("--push_yaw_magnitude", type=float, default=1.5, help="Yaw velocity impulse.")
parser.add_argument("--recovery_xy_error", type=float, default=0.45, help="Max xy velocity error for recovered state.")
parser.add_argument("--recovery_yaw_error", type=float, default=0.65, help="Max yaw velocity error for recovered state.")
parser.add_argument("--recovery_min_height", type=float, default=0.65, help="Min base height for recovered state.")
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
    env_cfg.scene.max_episode_length_s = max(20.0, args_cli.warmup + args_cli.duration + 2.0)
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
    root_pos_w = env._tensor(env.robot.data.root_pos_w)
    return root_lin_vel_yaw[:, 0], root_lin_vel_yaw[:, 1], root_ang_vel_w[:, 2], root_pos_w[:, 2]


def _build_env_and_policy():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    _configure_env(env_cfg, args_cli.num_envs)
    agent_cfg.load_run = args_cli.load_run
    agent_cfg.load_checkpoint = args_cli.checkpoint

    print("[INFO] Building push-recovery eval env", flush=True)
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
    return _zero_reset_reason_counts() if totals is None else dict(totals)


def _profile_velocity(profile: str, env, linear_magnitude: float, yaw_magnitude: float) -> torch.Tensor:
    vel_b = torch.zeros((env.num_envs, 6), device=env.device)
    if profile == "forward":
        vel_b[:, 0] = linear_magnitude
    elif profile == "backward":
        vel_b[:, 0] = -linear_magnitude
    elif profile == "left":
        vel_b[:, 1] = linear_magnitude
    elif profile == "right":
        vel_b[:, 1] = -linear_magnitude
    elif profile == "diag_left":
        vel_b[:, 0] = linear_magnitude
        vel_b[:, 1] = linear_magnitude
    elif profile == "diag_right":
        vel_b[:, 0] = linear_magnitude
        vel_b[:, 1] = -linear_magnitude
    elif profile == "yaw_left":
        vel_b[:, 5] = yaw_magnitude
    elif profile == "yaw_right":
        vel_b[:, 5] = -yaw_magnitude
    return vel_b


def _apply_push(env, profile: str):
    env_ids = torch.arange(env.num_envs, device=env.device)
    push_b = _profile_velocity(
        profile,
        env,
        float(args_cli.push_linear_magnitude),
        float(args_cli.push_yaw_magnitude),
    )
    root_quat_w = env._tensor(env.robot.data.root_quat_w)
    lin_vel_w = math_utils.quat_apply(math_utils.yaw_quat(root_quat_w), push_b[:, :3])
    vel_w = env._tensor(env.robot.data.root_vel_w).clone()
    vel_w[:, :3] += lin_vel_w
    vel_w[:, 5] += push_b[:, 5]
    env.robot.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def _run_one_command(env, policy, vx: float):
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset(env_ids)
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs, _ = obs
    if hasattr(env, "_eval_reset_reason_totals"):
        env._eval_reset_reason_totals = _zero_reset_reason_counts()

    warmup_steps = int(args_cli.warmup / env.step_dt)
    sample_steps = int(args_cli.duration / env.step_dt)
    push_interval_steps = max(1, int(args_cli.push_interval / env.step_dt))
    recovery_steps = max(1, int(args_cli.recovery_window / env.step_dt))

    pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    failed_since_push = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    push_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    profile_idx = 0

    vx_samples = []
    vy_samples = []
    wz_samples = []
    xy_err_samples = []
    height_samples = []
    resets = 0
    push_count = 0
    recovered_count = 0
    recovery_samples = []

    print(
        f"[INFO] Evaluating push recovery target_vx={vx:.3f} target_vy={args_cli.lin_vel_y:.3f} "
        f"target_wz={args_cli.ang_vel_z:.3f}",
        flush=True,
    )
    with torch.inference_mode():
        for step in range(warmup_steps + sample_steps):
            _force_command(env, vx, args_cli.lin_vel_y, args_cli.ang_vel_z)

            due = pending & ((step - push_step) >= recovery_steps)
            if bool(due.any().item()):
                root_vx, root_vy, root_wz, root_height = _root_command_state(env)
                xy_err = torch.sqrt((root_vx - vx) ** 2 + (root_vy - args_cli.lin_vel_y) ** 2)
                yaw_err = torch.abs(root_wz - args_cli.ang_vel_z)
                recovered = (
                    due
                    & (~failed_since_push)
                    & (root_height >= float(args_cli.recovery_min_height))
                    & (xy_err <= float(args_cli.recovery_xy_error))
                    & (yaw_err <= float(args_cli.recovery_yaw_error))
                )
                recovered_count += int(recovered.sum().item())
                recovery_samples.extend((step - push_step[due]).detach().float().mul(env.step_dt).cpu().tolist())
                pending[due] = False
                failed_since_push[due] = False
                push_step[due] = -1

            measured_step = step - warmup_steps
            should_push = step >= warmup_steps and measured_step % push_interval_steps == 0
            if should_push:
                profile = args_cli.profiles[profile_idx % len(args_cli.profiles)]
                profile_idx += 1
                _apply_push(env, profile)
                pending[:] = True
                failed_since_push[:] = False
                push_step[:] = step
                push_count += env.num_envs

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, tuple):
                obs, _ = obs
            if dones is not None:
                done_mask = dones.to(env.device).bool()
                resets += int(done_mask.sum().item())
                failed_since_push |= pending & done_mask

            _force_command(env, vx, args_cli.lin_vel_y, args_cli.ang_vel_z)
            root_vx, root_vy, root_wz, root_height = _root_command_state(env)
            vx_tensor = root_vx.detach().float()
            vy_tensor = root_vy.detach().float()
            wz_tensor = root_wz.detach().float()
            xy_err = torch.sqrt((vx_tensor - vx) ** 2 + (vy_tensor - args_cli.lin_vel_y) ** 2)
            yaw_err = torch.abs(wz_tensor - args_cli.ang_vel_z)

            if step >= warmup_steps:
                vx_samples.extend(vx_tensor.cpu().tolist())
                vy_samples.extend(vy_tensor.cpu().tolist())
                wz_samples.extend(wz_tensor.cpu().tolist())
                xy_err_samples.extend(xy_err.cpu().tolist())
                height_samples.extend(root_height.detach().float().cpu().tolist())

    unresolved = int(pending.sum().item())
    reset_reasons = _reset_reason_counts(env)
    abs_samples = sorted(abs(value) for value in vx_samples) if vx_samples else [0.0]
    sorted_xy_err = sorted(xy_err_samples) if xy_err_samples else [0.0]
    sorted_height = sorted(height_samples) if height_samples else [0.0]
    p90_idx = int(0.9 * (len(abs_samples) - 1))
    p10_idx = int(0.1 * (len(sorted_height) - 1))
    resolved_pushes = max(push_count - unresolved, 1)
    return {
        "target": vx,
        "target_vy": args_cli.lin_vel_y,
        "target_wz": args_cli.ang_vel_z,
        "push_linear": float(args_cli.push_linear_magnitude),
        "push_yaw": float(args_cli.push_yaw_magnitude),
        "push_count": push_count,
        "recovered": recovered_count,
        "recovery_ratio": recovered_count / resolved_pushes,
        "unresolved_pushes": unresolved,
        "mean_recovery_s": statistics.fmean(recovery_samples) if recovery_samples else 0.0,
        "mean_vx": statistics.fmean(vx_samples) if vx_samples else 0.0,
        "mean_vy": statistics.fmean(vy_samples) if vy_samples else 0.0,
        "mean_wz": statistics.fmean(wz_samples) if wz_samples else 0.0,
        "xy_abs_err": statistics.fmean(xy_err_samples) if xy_err_samples else 0.0,
        "p90_abs_vx": abs_samples[p90_idx],
        "p90_xy_err": sorted_xy_err[p90_idx],
        "p10_height": sorted_height[p10_idx],
        "resets": resets,
        **reset_reasons,
    }


def main():
    print("[INFO] Push-recovery eval start", flush=True)
    env, policy = _build_env_and_policy()
    try:
        for speed in args_cli.speeds:
            result = _run_one_command(env, policy, speed)
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
