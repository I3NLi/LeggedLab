import argparse
import os
import re
import statistics

import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate gait quality metrics for a trained locomotion policy.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--load_run", type=str, required=True, help="Run folder to load from.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint filename to load.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--duration", type=float, default=8.0, help="Measured duration per speed in seconds.")
parser.add_argument("--warmup", type=float, default=3.0, help="Warmup duration before measuring.")
parser.add_argument("--speeds", nargs="+", type=float, required=True, help="Fixed x velocity commands to evaluate.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from legged_lab.envs import *  # noqa: E402,F401,F403
from legged_lab.utils import task_registry  # noqa: E402


def _configure_env(env_cfg, num_envs: int):
    env_cfg.noise.add_noise = False
    env_cfg.noise.add_bias = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.episode_length_curriculum.enable = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
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


def _actor_tensor(obs):
    if isinstance(obs, tuple):
        obs, _ = obs
    if isinstance(obs, dict) or hasattr(obs, "keys"):
        return obs["actor"]
    return obs


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


def _resolve_robot_bodies(env, body_names):
    cfg = SceneEntityCfg("robot", body_names=body_names)
    cfg.resolve(env.scene)
    return list(cfg.body_ids)


def _joint_ids(env, patterns):
    names = list(env.robot.joint_names if hasattr(env.robot, "joint_names") else env.robot.data.joint_names)
    regexes = [re.compile(pattern) for pattern in patterns]
    return [idx for idx, name in enumerate(names) if any(regex.search(name) for regex in regexes)]


def _mean_or_zero(values):
    return statistics.fmean(values) if values else 0.0


def _percentile(values, q):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(q * (len(sorted_values) - 1))
    return sorted_values[idx]


def _build_env_and_policy():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    _configure_env(env_cfg, args_cli.num_envs)
    agent_cfg.load_run = args_cli.load_run
    agent_cfg.load_checkpoint = args_cli.checkpoint

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)
    env.command_metrics_enabled = False
    if hasattr(env.robot, "root_view"):
        env.robot.data._root_view = env.robot.root_view

    log_root_path = os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=os.path.dirname(resume_path), device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    foot_body_ids = _resolve_robot_bodies(env, env.cfg.robot.feet_body_names)
    arm_joint_ids = _joint_ids(env, [r"_shoulder_", r"_elbow_", r"_wrist_"])
    shoulder_pitch_ids = _joint_ids(env, [r"_shoulder_pitch_joint"])

    print(
        "[INFO] Metric ids: "
        f"foot_body_ids={foot_body_ids}, "
        f"arm_joint_count={len(arm_joint_ids)}, "
        f"shoulder_pitch_count={len(shoulder_pitch_ids)}",
        flush=True,
    )
    return env, policy, foot_body_ids, arm_joint_ids, shoulder_pitch_ids


def _run_one_speed(env, policy, foot_body_ids, arm_joint_ids, shoulder_pitch_ids, vx: float):
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset(env_ids)
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs, _ = obs

    warmup_steps = int(args_cli.warmup / env.step_dt)
    sample_steps = int(args_cli.duration / env.step_dt)
    resets = 0
    transition_count = 0
    previous_contacts = None

    vx_samples = []
    tilt_xy_samples = []
    gravity_x_samples = []
    foot_z_samples = []
    swing_foot_z_samples = []
    contact_ratio_samples = []
    single_stance_samples = []
    double_stance_samples = []
    flight_samples = []
    arm_l1_samples = []
    shoulder_pitch_samples = []

    command_dim = 4
    joint_pos_start = 6 + command_dim
    joint_pos_end = joint_pos_start + env.num_actions

    print(f"[INFO] Evaluating gait target={vx:.3f}", flush=True)
    with torch.inference_mode():
        for step in range(warmup_steps + sample_steps):
            _force_command(env, vx)
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            if isinstance(obs, tuple):
                obs, _ = obs
            if dones is not None:
                resets += int(dones.sum().item())
            _force_command(env, vx)

            if step < warmup_steps:
                continue

            actor_obs = _actor_tensor(obs)
            current_actor_obs = actor_obs[:, :joint_pos_end]
            projected_gravity = current_actor_obs[:, 3:6] / env.obs_scales.projected_gravity
            joint_offsets = current_actor_obs[:, joint_pos_start:joint_pos_end] / env.obs_scales.joint_pos

            vx_samples.extend(_critic_body_frame_vx(env, extras).detach().float().cpu().tolist())
            tilt_xy_samples.extend(torch.norm(projected_gravity[:, :2], dim=1).detach().float().cpu().tolist())
            gravity_x_samples.extend(projected_gravity[:, 0].detach().float().cpu().tolist())

            net_contact = torch.as_tensor(env.contact_sensor.data.net_forces_w_history, device=env.device)
            foot_contacts = torch.max(
                torch.norm(net_contact[:, :, env.feet_cfg.body_ids], dim=-1), dim=1
            )[0] > 1.0
            contact_counts = foot_contacts.sum(dim=1)
            contact_ratio_samples.extend(foot_contacts.float().mean(dim=1).detach().float().cpu().tolist())
            single_stance_samples.extend((contact_counts == 1).float().detach().cpu().tolist())
            double_stance_samples.extend((contact_counts == 2).float().detach().cpu().tolist())
            flight_samples.extend((contact_counts == 0).float().detach().cpu().tolist())
            if previous_contacts is not None:
                transition_count += int((foot_contacts != previous_contacts).sum().item())
            previous_contacts = foot_contacts.clone()

            body_pos_w = torch.as_tensor(env.robot.data.body_pos_w, device=env.device)
            foot_z = body_pos_w[:, foot_body_ids, 2]
            foot_z_samples.extend(foot_z.reshape(-1).detach().float().cpu().tolist())
            if torch.any(~foot_contacts):
                swing_foot_z_samples.extend(foot_z[~foot_contacts].detach().float().cpu().tolist())

            if arm_joint_ids:
                arm_l1_samples.extend(torch.mean(torch.abs(joint_offsets[:, arm_joint_ids]), dim=1).cpu().tolist())
            if shoulder_pitch_ids:
                shoulder_pitch_samples.extend(
                    torch.mean(torch.abs(joint_offsets[:, shoulder_pitch_ids]), dim=1).cpu().tolist()
                )

    return {
        "target": vx,
        "mean_vx": _mean_or_zero(vx_samples),
        "abs_err": _mean_or_zero([abs(value - vx) for value in vx_samples]),
        "p50_vx": statistics.median(vx_samples) if vx_samples else 0.0,
        "p90_abs_vx": _percentile([abs(value) for value in vx_samples], 0.90),
        "resets": resets,
        "mean_tilt_xy": _mean_or_zero(tilt_xy_samples),
        "mean_gravity_x": _mean_or_zero(gravity_x_samples),
        "mean_foot_z": _mean_or_zero(foot_z_samples),
        "p90_swing_foot_z": _percentile(swing_foot_z_samples, 0.90),
        "contact_ratio": _mean_or_zero(contact_ratio_samples),
        "single_stance_ratio": _mean_or_zero(single_stance_samples),
        "double_stance_ratio": _mean_or_zero(double_stance_samples),
        "flight_ratio": _mean_or_zero(flight_samples),
        "contact_transitions_per_env_s": transition_count / max(args_cli.duration * env.num_envs, 1e-6),
        "mean_arm_joint_abs_offset": _mean_or_zero(arm_l1_samples),
        "mean_shoulder_pitch_abs_offset": _mean_or_zero(shoulder_pitch_samples),
    }


def main():
    env, policy, foot_body_ids, arm_joint_ids, shoulder_pitch_ids = _build_env_and_policy()
    try:
        for speed in args_cli.speeds:
            result = _run_one_speed(env, policy, foot_body_ids, arm_joint_ids, shoulder_pitch_ids, speed)
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
