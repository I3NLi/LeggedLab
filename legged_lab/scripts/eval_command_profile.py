import argparse
import os
import statistics

import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a policy under a staged forward-speed command profile.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--load_run", type=str, required=True, help="Run folder to load from.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint filename to load.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--warmup", type=float, default=1.0, help="Warmup duration before the profile starts.")
parser.add_argument("--targets", nargs="+", type=float, required=True, help="Forward speed targets for each segment.")
parser.add_argument("--durations", nargs="+", type=float, required=True, help="Duration for each segment in seconds.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.utils import task_registry  # noqa: E402


def _configure_env(env_cfg, num_envs: int, total_duration: float):
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.episode_length_curriculum.enable = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.max_episode_length_s = max(20.0, total_duration + 5.0)
    env_cfg.scene.seed = 42

    if hasattr(env_cfg.scene, "height_scanner"):
        env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)


def _force_command(env, vx: float):
    target = torch.zeros((env.num_envs, 3), device=env.device)
    target[:, 0] = vx
    env.command_generator.vel_command_b[:] = target
    env.command_generator.is_standing_env[:] = False
    env.command_generator.is_heading_env[:] = False
    if hasattr(env.command_generator, "time_left"):
        env.command_generator.time_left[:] = 1e6
    if getattr(env, "_command_slew_enabled", False):
        env._command_buf[:] = target


def _critic_body_frame_vx(env, extras):
    critic_obs = extras.get("observations", {}).get("critic")
    if critic_obs is None:
        raise RuntimeError("Critic observations are missing from env extras; cannot read root velocity.")
    feet_count = len(env.feet_cfg.body_ids)
    root_lin_vel_start = critic_obs.shape[-1] - feet_count - 3
    return critic_obs[:, root_lin_vel_start] / env.obs_scales.lin_vel


def _actor_tensor(obs):
    if isinstance(obs, tuple):
        obs, _ = obs
    if isinstance(obs, dict) or hasattr(obs, "keys"):
        return obs["actor"]
    return obs


def _zero_reset_reason_counts():
    return {
        "timeout_resets": 0,
        "head_shoulder_resets": 0,
        "body_contact_resets": 0,
        "speed_tracking_resets": 0,
        "other_resets": 0,
    }


def _install_reset_reason_counter(env):
    if not hasattr(env, "_log_reset_reasons") or getattr(env, "_profile_reset_reason_counter_installed", False):
        return

    original_log_reset_reasons = env._log_reset_reasons
    env._profile_reset_reason_totals = _zero_reset_reason_counts()

    def _counting_log_reset_reasons(env_ids):
        original_log_reset_reasons(env_ids)
        totals = getattr(env, "_profile_reset_reason_totals", None)
        if totals is None:
            return
        log = env.extras.get("log", {})
        totals["timeout_resets"] += int(round(float(log.get("Reset/timeout_count", 0.0))))
        totals["head_shoulder_resets"] += int(round(float(log.get("Reset/head_shoulder_contact_count", 0.0))))
        totals["body_contact_resets"] += int(round(float(log.get("Reset/body_contact_count", 0.0))))
        totals["speed_tracking_resets"] += int(round(float(log.get("Reset/speed_tracking_failure_count", 0.0))))
        totals["other_resets"] += int(round(float(log.get("Reset/other_count", 0.0))))

    env._log_reset_reasons = _counting_log_reset_reasons
    env._profile_reset_reason_counter_installed = True


def _reset_reason_counts(env):
    return dict(getattr(env, "_profile_reset_reason_totals", _zero_reset_reason_counts()))


def _reset_reason_delta(before, after):
    return {key: after[key] - before.get(key, 0) for key in after}


def _mean_or_zero(values):
    return statistics.fmean(values) if values else 0.0


def _percentile(values, q):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(q * (len(sorted_values) - 1))
    return sorted_values[idx]


def _build_env_and_policy():
    if len(args_cli.targets) != len(args_cli.durations):
        raise ValueError("--targets and --durations must have the same length.")

    total_duration = args_cli.warmup + sum(args_cli.durations)
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    _configure_env(env_cfg, args_cli.num_envs, total_duration)
    agent_cfg.load_run = args_cli.load_run
    agent_cfg.load_checkpoint = args_cli.checkpoint

    print("[INFO] Building command-profile eval env", flush=True)
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


def _sample_state(env, extras, obs):
    vx = _critic_body_frame_vx(env, extras).detach().float()
    actor_obs = _actor_tensor(obs)
    projected_gravity = actor_obs[:, 3:6] / env.obs_scales.projected_gravity
    tilt_xy = torch.norm(projected_gravity[:, :2], dim=1).detach().float()
    return vx, tilt_xy


def _run_profile(env, policy):
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset(env_ids)
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs, _ = obs
    if hasattr(env, "_profile_reset_reason_totals"):
        env._profile_reset_reason_totals = _zero_reset_reason_counts()

    print(f"[INFO] Warmup target={args_cli.targets[0]:.3f} duration={args_cli.warmup:.3f}", flush=True)
    with torch.inference_mode():
        for _ in range(int(args_cli.warmup / env.step_dt)):
            _force_command(env, args_cli.targets[0])
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if isinstance(obs, tuple):
                obs, _ = obs
            _force_command(env, args_cli.targets[0])

        results = []
        for segment_idx, (target_vx, duration) in enumerate(zip(args_cli.targets, args_cli.durations)):
            steps = int(duration / env.step_dt)
            first_window_steps = min(steps, max(1, int(1.0 / env.step_dt)))
            last_window_start = max(0, steps - first_window_steps)
            before_reasons = _reset_reason_counts(env)
            vx_samples = []
            first_vx_samples = []
            last_vx_samples = []
            tilt_xy_samples = []
            resets = 0

            print(
                f"[INFO] Segment {segment_idx} target={target_vx:.3f} duration={duration:.3f}",
                flush=True,
            )
            for step in range(steps):
                _force_command(env, target_vx)
                actions = policy(obs)
                obs, _, dones, extras = env.step(actions)
                if isinstance(obs, tuple):
                    obs, _ = obs
                if dones is not None:
                    resets += int(dones.sum().item())
                _force_command(env, target_vx)

                vx, tilt_xy = _sample_state(env, extras, obs)
                vx_values = vx.cpu().tolist()
                vx_samples.extend(vx_values)
                if step < first_window_steps:
                    first_vx_samples.extend(vx_values)
                if step >= last_window_start:
                    last_vx_samples.extend(vx_values)
                tilt_xy_samples.extend(tilt_xy.cpu().tolist())

            reason_counts = _reset_reason_delta(before_reasons, _reset_reason_counts(env))
            result = {
                "segment": segment_idx,
                "target": target_vx,
                "duration": duration,
                "mean_vx": _mean_or_zero(vx_samples),
                "first1s_mean_vx": _mean_or_zero(first_vx_samples),
                "last1s_mean_vx": _mean_or_zero(last_vx_samples),
                "abs_err": _mean_or_zero([abs(value - target_vx) for value in vx_samples]),
                "p90_abs_vx": _percentile([abs(value) for value in vx_samples], 0.90),
                "mean_tilt_xy": _mean_or_zero(tilt_xy_samples),
                "p90_tilt_xy": _percentile(tilt_xy_samples, 0.90),
                "resets": resets,
                **reason_counts,
            }
            results.append(result)
            print(
                "RESULT "
                + " ".join(
                    f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in result.items()
                ),
                flush=True,
            )
    return results


def main():
    env, policy = _build_env_and_policy()
    try:
        _run_profile(env, policy)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
