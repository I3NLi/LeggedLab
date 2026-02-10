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

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from rsl_rl.env import VecEnv

from legged_lab.envs.base.base_env_config import BaseEnvCfg
from legged_lab.utils.env_utils.scene import SceneCfg


class BaseEnv(VecEnv):
    def __init__(self, cfg: BaseEnvCfg, headless):
        self.cfg: BaseEnvCfg

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_reward_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._episode_had_teleport_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._teleported_this_step_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._termination_contact_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._termination_contact_delay_s = max(0.0, float(self.cfg.robot.terminate_contacts_delay_s))
        self._static_log_info = {
            "Config/termination_delay_s": float(self._termination_contact_delay_s),
        }
        self._episode_len_curriculum_round = 0
        self._episode_len_curriculum_sum = 0.0
        self._episode_len_curriculum_count = 0
        self._episode_len_curriculum_streak = 0
        self._episode_len_curriculum_updates = 0
        self._episode_len_curriculum_last_round_mean = 0.0
        self._episode_reward_curriculum_sum = 0.0
        self._episode_reward_curriculum_last_round_mean = 0.0
        self.init_obs_buffer()

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
            ],
            dim=-1,
        )

        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1
        )

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        self.extras["log"].update(self._static_log_info)
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self._update_episode_length_curriculum(env_ids)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.episode_reward_buf[env_ids] = 0.0
        self._episode_had_teleport_buf[env_ids] = False
        self._teleported_this_step_buf[env_ids] = False
        self._termination_contact_time_buf[env_ids] = 0.0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):

        delayed_actions = self.action_buffer.compute(actions)

        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self._teleported_this_step_buf[:] = False
        teleported_count = self._teleport_out_of_bounds_envs()
        if teleported_count > 0:
            self.extras.setdefault("log", dict())["Env/out_of_bounds_teleports"] = float(teleported_count)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.episode_reward_buf += reward_buf
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        termination_contact = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        # Ignore contact-based termination only in the step where teleport happened.
        # This prevents teleport itself from causing an immediate reset, while future falls still terminate early.
        termination_contact &= ~self._teleported_this_step_buf
        if self._termination_contact_delay_s <= 0.0:
            reset_buf = termination_contact
        else:
            # Accumulate continuous fall duration; reset when contact clears.
            contact_mask = termination_contact.to(self._termination_contact_time_buf.dtype)
            self._termination_contact_time_buf.mul_(contact_mask).add_(contact_mask * self.step_dt)
            reset_buf = self._termination_contact_time_buf >= self._termination_contact_delay_s
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def _teleport_out_of_bounds_envs(self) -> int:
        terrain_generator = self.cfg.scene.terrain_generator
        if terrain_generator is None:
            return 0
        if not self.cfg.scene.out_of_bounds_teleport_enable:
            return 0

        terrain_size_x, terrain_size_y = terrain_generator.size
        half_size_x = float(terrain_size_x) * 0.5
        half_size_y = float(terrain_size_y) * 0.5
        trigger_scale = max(float(self.cfg.scene.out_of_bounds_teleport_trigger_scale), 0.0)
        trigger_half_x = half_size_x * trigger_scale
        trigger_half_y = half_size_y * trigger_scale
        root_xy = self.robot.data.root_pos_w[:, :2]
        rel_xy = root_xy - self.scene.env_origins[:, :2]

        out_of_bounds = (torch.abs(rel_xy[:, 0]) > trigger_half_x) | (torch.abs(rel_xy[:, 1]) > trigger_half_y)
        env_ids = out_of_bounds.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return 0

        safe_margin = max(float(self.cfg.scene.out_of_bounds_teleport_margin), 0.0)
        spawn_half_x = max(half_size_x - safe_margin, 0.0)
        spawn_half_y = max(half_size_y - safe_margin, 0.0)
        offsets = torch.zeros((env_ids.numel(), 2), device=self.device)
        if spawn_half_x > 0.0:
            offsets[:, 0] = (torch.rand(env_ids.numel(), device=self.device) * 2.0 - 1.0) * spawn_half_x
        if spawn_half_y > 0.0:
            offsets[:, 1] = (torch.rand(env_ids.numel(), device=self.device) * 2.0 - 1.0) * spawn_half_y

        # Only relocate x/y. Root orientation, joint posture, and velocities are preserved.
        root_state = self.robot.data.root_state_w[env_ids].clone()
        root_state[:, :2] = self.scene.env_origins[env_ids, :2] + offsets
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self._episode_had_teleport_buf[env_ids] = True
        self._teleported_this_step_buf[env_ids] = True
        return int(env_ids.numel())

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6:9] = 0
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def _update_episode_length_curriculum(self, env_ids):
        cfg = self.cfg.episode_length_curriculum
        if not cfg.enable:
            return

        round_completed = False
        speed_updated = False
        required_streak_rounds = max(int(cfg.required_streak_rounds), 1)
        target_episode_count = cfg.round_episode_count if cfg.round_episode_count > 0 else self.num_envs

        episode_lengths = self.episode_length_buf[env_ids].float()
        valid_mask = episode_lengths > 0.0
        # Only count episodes that survived to timeout.
        valid_mask &= self.time_out_buf[env_ids]
        # Ignore episodes that used out-of-bounds teleport so curriculum mean episode length is not biased.
        valid_mask &= ~self._episode_had_teleport_buf[env_ids]
        valid_episode_lengths = episode_lengths[valid_mask]
        episode_rewards = self.episode_reward_buf[env_ids].float()
        valid_episode_rewards = episode_rewards[valid_mask]
        if valid_episode_lengths.numel() > 0:
            self._episode_len_curriculum_sum += float(valid_episode_lengths.sum().item())
            self._episode_len_curriculum_count += int(valid_episode_lengths.numel())
            self._episode_reward_curriculum_sum += float(valid_episode_rewards.sum().item())

            if self._episode_len_curriculum_count >= target_episode_count:
                round_completed = True
                self._episode_len_curriculum_round += 1
                self._episode_len_curriculum_last_round_mean = self._episode_len_curriculum_sum / max(
                    self._episode_len_curriculum_count, 1
                )
                self._episode_reward_curriculum_last_round_mean = self._episode_reward_curriculum_sum / max(
                    self._episode_len_curriculum_count, 1
                )
                round_threshold = float(self.max_episode_length) * float(cfg.episode_length_ratio)
                reward_threshold = float(cfg.min_mean_reward)
                if (
                    self._episode_len_curriculum_last_round_mean >= round_threshold
                    and self._episode_reward_curriculum_last_round_mean >= reward_threshold
                ):
                    self._episode_len_curriculum_streak += 1
                else:
                    self._episode_len_curriculum_streak = 0

                self._episode_len_curriculum_sum = 0.0
                self._episode_len_curriculum_count = 0
                self._episode_reward_curriculum_sum = 0.0

                if self._episode_len_curriculum_streak >= required_streak_rounds:
                    x_min, x_max = self.command_generator.cfg.ranges.lin_vel_x
                    if float(cfg.max_forward_speed) > 0.0 and np.isfinite(float(cfg.max_forward_speed)):
                        new_x_max = min(float(x_max) + float(cfg.speed_increment), float(cfg.max_forward_speed))
                    else:
                        new_x_max = float(x_max) + float(cfg.speed_increment)

                    if new_x_max > float(x_max):
                        new_x_range = (float(x_min), float(new_x_max))
                        self.command_generator.cfg.ranges.lin_vel_x = new_x_range
                        self.cfg.commands.ranges.lin_vel_x = new_x_range
                        self._episode_len_curriculum_updates += 1
                        speed_updated = True

                    self._episode_len_curriculum_streak = 0

        x_min, x_max = self.command_generator.cfg.ranges.lin_vel_x
        self.extras["log"].update(
            {
                "Curriculum/round_index": float(self._episode_len_curriculum_round),
                "Curriculum/round_mean_episode_length": float(self._episode_len_curriculum_last_round_mean),
                "Curriculum/round_mean_episode_reward": float(self._episode_reward_curriculum_last_round_mean),
                "Curriculum/episode_length_streak": float(self._episode_len_curriculum_streak),
                "Curriculum/forward_speed_min": float(x_min),
                "Curriculum/forward_speed_max": float(x_max),
                "Curriculum/speed_updates": float(self._episode_len_curriculum_updates),
            }
        )

        if cfg.print_status and round_completed:
            step = self.sim_step_counter // self.cfg.sim.decimation
            cap_text = (
                f"{float(cfg.max_forward_speed):.3f}"
                if float(cfg.max_forward_speed) > 0.0 and np.isfinite(float(cfg.max_forward_speed))
                else "inf"
            )
            update_tag = " [speed+]" if speed_updated else ""
            print(
                "[CURRICULUM][LeggedLab] "
                f"step={step} round={self._episode_len_curriculum_round} "
                f"mean_ep_len={self._episode_len_curriculum_last_round_mean:.2f}/{float(self.max_episode_length):.2f} "
                f"mean_ep_rew={self._episode_reward_curriculum_last_round_mean:.2f}/{float(cfg.min_mean_reward):.2f} "
                f"streak={self._episode_len_curriculum_streak}/{required_streak_rounds} "
                f"lin_vel_x=({float(x_min):.3f}, {float(x_max):.3f}) "
                f"cap={cap_text} updates={self._episode_len_curriculum_updates}{update_tag}"
            )

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
