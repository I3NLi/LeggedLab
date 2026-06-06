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
import isaaclab.utils.math as math_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import inspect
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

try:
    from tensordict import TensorDict
except ImportError:
    TensorDict = None


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

        sim_cfg_kwargs = {
            "device": cfg.device,
            "dt": cfg.sim.dt,
            "render_interval": cfg.sim.decimation,
            "physics_material": sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        }
        physics_cfg = PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count)
        simulation_cfg_params = inspect.signature(sim_utils.SimulationCfg).parameters
        if "physics" in simulation_cfg_params:
            sim_cfg_kwargs["physics"] = physics_cfg
        else:
            sim_cfg_kwargs["physx"] = physics_cfg
        sim_cfg = sim_utils.SimulationCfg(
            **sim_cfg_kwargs,
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
            debug_vis=False,
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
        self.immediate_termination_contact_cfg = None
        if self.cfg.robot.immediate_terminate_contacts_body_names:
            self.immediate_termination_contact_cfg = SceneEntityCfg(
                name="contact_sensor", body_names=self.cfg.robot.immediate_terminate_contacts_body_names
            )
            self.immediate_termination_contact_cfg.resolve(self.scene)
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
        self._stuck_command_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self._stuck_command_reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._termination_contact_delay_s = max(0.0, float(self.cfg.robot.terminate_contacts_delay_s))
        self._termination_contact_enabled = True
        self._static_log_info = {
            "Config/termination_delay_s": float(self._termination_contact_delay_s),
        }
        self._episode_len_curriculum_round = 0
        self._episode_len_curriculum_sum = 0.0
        self._episode_len_curriculum_count = 0
        self._episode_len_curriculum_timeout_count = 0
        self._episode_len_curriculum_streak = 0
        self._episode_len_curriculum_stage_successes = 0
        self._episode_len_curriculum_speed_updates = 0
        self._episode_len_curriculum_stage_idx = -1
        self._episode_len_curriculum_last_round_mean = 0.0
        self._episode_len_curriculum_last_round_timeout_ratio = 0.0
        self._episode_len_curriculum_last_round_success = False
        self._episode_len_curriculum_last_round_reward_success = False
        self._episode_len_curriculum_last_round_timeout_success = False
        self._episode_reward_curriculum_sum = 0.0
        self._episode_reward_curriculum_last_round_mean = 0.0
        self._rsl_rl_uses_tensordict_obs = self._detect_rsl_rl_tensordict_observations()
        self._velocity_debug_draw = None
        self._velocity_debug_vis_enabled = bool(self.cfg.commands.debug_vis and not self.headless)
        if self._velocity_debug_vis_enabled:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

            self._velocity_debug_draw = omni_debug_draw.acquire_debug_draw_interface()
        self.init_obs_buffer()

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = self._tensor(robot.data.root_ang_vel_b)
        projected_gravity = self._tensor(robot.data.projected_gravity_b)
        command = self._command_tensor()
        joint_pos = self._tensor(robot.data.joint_pos) - self._tensor(robot.data.default_joint_pos)
        joint_vel = self._tensor(robot.data.joint_vel) - self._tensor(robot.data.default_joint_vel)
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

        root_lin_vel = self._tensor(robot.data.root_lin_vel_b)
        net_contact_forces = self._tensor(net_contact_forces)
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1
        )

        return current_actor_obs, current_critic_obs

    def _command_tensor(self):
        return self._tensor(self.command_generator.command)

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

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
        self._stuck_command_time_buf[env_ids] = 0.0
        self._stuck_command_reset_buf[env_ids] = False

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):

        delayed_actions = self.action_buffer.compute(actions)

        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale + self._tensor(self.robot.data.default_joint_pos)

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
        self._draw_velocity_debug_arrows()
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

        if self._rsl_rl_uses_tensordict_obs:
            return self._obs_tensor_dict(actor_obs, critic_obs), reward_buf, self.reset_buf, self.extras
        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self._tensor(self.contact_sensor.data.net_forces_w_history)
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        self._stuck_command_reset_buf[:] = False

        immediate_termination_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.immediate_termination_contact_cfg is not None:
            immediate_termination_contact = torch.any(
                torch.max(
                    torch.norm(
                        net_contact_forces[:, :, self.immediate_termination_contact_cfg.body_ids],
                        dim=-1,
                    ),
                    dim=1,
                )[0]
                > 1.0,
                dim=1,
            )
        immediate_termination_contact &= ~self._teleported_this_step_buf

        if not self._termination_contact_enabled:
            reset_buf = immediate_termination_contact | self._compute_stuck_command_reset() | time_out_buf
            return reset_buf, time_out_buf

        delayed_termination_contact = torch.any(
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
        delayed_termination_contact &= ~self._teleported_this_step_buf
        if self._termination_contact_delay_s <= 0.0:
            reset_buf = delayed_termination_contact
        else:
            # Accumulate continuous fall duration; reset when contact clears.
            contact_mask = delayed_termination_contact.to(self._termination_contact_time_buf.dtype)
            self._termination_contact_time_buf.mul_(contact_mask).add_(contact_mask * self.step_dt)
            reset_buf = self._termination_contact_time_buf >= self._termination_contact_delay_s
        reset_buf |= immediate_termination_contact
        reset_buf |= self._compute_stuck_command_reset()
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def _compute_stuck_command_reset(self) -> torch.Tensor:
        if not self.cfg.robot.terminate_when_stuck:
            return self._stuck_command_reset_buf

        command = self._command_tensor()
        command_speed = torch.norm(command[:, :2], dim=1)
        root_speed = torch.norm(self._tensor(self.robot.data.root_lin_vel_w)[:, :2], dim=1)
        past_grace = self.episode_length_buf * self.step_dt >= max(0.0, float(self.cfg.robot.stuck_grace_s))
        stuck = (
            (command_speed > float(self.cfg.robot.stuck_command_threshold))
            & (root_speed < float(self.cfg.robot.stuck_speed_threshold))
            & past_grace
            & ~self._teleported_this_step_buf
        )
        contact_mask = stuck.to(self._stuck_command_time_buf.dtype)
        self._stuck_command_time_buf.mul_(contact_mask).add_(contact_mask * self.step_dt)
        self._stuck_command_reset_buf[:] = self._stuck_command_time_buf >= max(
            0.0, float(self.cfg.robot.stuck_duration_s)
        )
        return self._stuck_command_reset_buf

    def _draw_velocity_debug_arrows(self):
        if self._velocity_debug_draw is None:
            return

        self._velocity_debug_draw.clear_lines()
        root_pos = self._tensor(self.robot.data.root_pos_w).clone()
        root_pos[:, 2] += 0.7
        command = self._command_tensor()
        command_3d = torch.cat([command[:, :2], torch.zeros_like(command[:, :1])], dim=1)
        command_w = math_utils.quat_apply_yaw(self._tensor(self.robot.data.root_quat_w), command_3d)
        actual_w = self._tensor(self.robot.data.root_lin_vel_w)

        starts = []
        ends = []
        colors = []
        thicknesses = []
        for velocity, color in (
            (command_w, (0.0, 1.0, 0.0, 1.0)),
            (actual_w, (0.0, 0.35, 1.0, 1.0)),
        ):
            arrow_vec = velocity[:, :3].clone()
            arrow_vec[:, 2] = 0.0
            arrow_len = torch.norm(arrow_vec[:, :2], dim=1, keepdim=True).clamp(min=1.0e-6)
            direction = arrow_vec / arrow_len
            scale = torch.clamp(arrow_len, max=2.5) * 0.35
            end = root_pos + direction * scale

            starts.extend(root_pos.tolist())
            ends.extend(end.tolist())
            colors.extend([color] * self.num_envs)
            thicknesses.extend([4.0] * self.num_envs)

            side = torch.stack([-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])], dim=1)
            head_base = end - direction * 0.18
            left = head_base + side * 0.08
            right = head_base - side * 0.08
            starts.extend(end.tolist())
            ends.extend(left.tolist())
            starts.extend(end.tolist())
            ends.extend(right.tolist())
            colors.extend([color] * (2 * self.num_envs))
            thicknesses.extend([4.0] * (2 * self.num_envs))

        self._velocity_debug_draw.draw_lines(starts, ends, colors, thicknesses)

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
        root_xy = self._tensor(self.robot.data.root_pos_w)[:, :2]
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
        root_state = self._tensor(self.robot.data.root_state_w)[env_ids].clone()
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

        def _resolve_stage(successes: int):
            stage_cfg = None
            stage_idx = -1
            stage_max_updates = -1
            stage_progress = 0
            if getattr(cfg, "stages", None):
                remaining = max(int(successes), 0)
                consumed = 0
                for idx, stage in enumerate(cfg.stages):
                    max_updates = int(stage.max_updates) if stage.max_updates is not None else -1
                    if max_updates < 0:
                        stage_cfg = stage
                        stage_idx = idx
                        stage_max_updates = -1
                        stage_progress = remaining
                        break
                    if remaining < max_updates:
                        stage_cfg = stage
                        stage_idx = idx
                        stage_max_updates = max_updates
                        stage_progress = remaining
                        break
                    remaining -= max_updates
                    consumed += max_updates
                if stage_cfg is None:
                    stage_cfg = cfg.stages[-1]
                    stage_idx = len(cfg.stages) - 1
                    stage_max_updates = int(stage_cfg.max_updates) if stage_cfg.max_updates is not None else -1
                    stage_progress = max(int(successes) - consumed, 0)
            return stage_cfg, stage_idx, stage_max_updates, stage_progress

        def _apply_reward_weight(term_name: str, weight_value: float | None):
            if weight_value is None:
                return
            weight_value = float(weight_value)
            if hasattr(self.cfg.reward, term_name):
                getattr(self.cfg.reward, term_name).weight = weight_value
            try:
                term_cfg = self.reward_manager.get_term_cfg(term_name)
            except ValueError:
                return
            term_cfg.weight = weight_value
            self.reward_manager.set_term_cfg(term_name, term_cfg)

        def _apply_stage_overrides(stage_cfg):
            if stage_cfg is None:
                return

            def _apply_range(attr_name, value):
                if value is None:
                    return
                new_range = (float(value[0]), float(value[1]))
                setattr(self.command_generator.cfg.ranges, attr_name, new_range)
                setattr(self.cfg.commands.ranges, attr_name, new_range)

            _apply_range("lin_vel_x", stage_cfg.lin_vel_x)
            _apply_range("lin_vel_y", stage_cfg.lin_vel_y)
            _apply_range("ang_vel_z", stage_cfg.ang_vel_z)

            if stage_cfg.termination_contact_enabled is not None:
                self._termination_contact_enabled = bool(stage_cfg.termination_contact_enabled)
                # Reset accumulated fall time to avoid stale termination when toggling modes.
                self._termination_contact_time_buf.zero_()
            if stage_cfg.termination_contact_delay_s is not None:
                self._termination_contact_delay_s = max(0.0, float(stage_cfg.termination_contact_delay_s))
            if stage_cfg.reset_joint_pos_range is not None:
                new_pos_range = (
                    float(stage_cfg.reset_joint_pos_range[0]),
                    float(stage_cfg.reset_joint_pos_range[1]),
                )
                self.cfg.domain_rand.events.reset_robot_joints.params["position_range"] = new_pos_range
                # Keep EventManager runtime config in sync when supported.
                if hasattr(self.event_manager, "get_term_cfg") and hasattr(self.event_manager, "set_term_cfg"):
                    try:
                        term_cfg = self.event_manager.get_term_cfg("reset_robot_joints")
                        term_cfg.params["position_range"] = new_pos_range
                        self.event_manager.set_term_cfg("reset_robot_joints", term_cfg)
                    except Exception:
                        pass

            _apply_reward_weight("track_lin_vel_xy_exp", stage_cfg.track_lin_vel_xy_exp_weight)
            _apply_reward_weight("track_ang_vel_z_exp", stage_cfg.track_ang_vel_z_exp_weight)
            _apply_reward_weight("feet_air_time", stage_cfg.feet_air_time_weight)
            _apply_reward_weight("energy", stage_cfg.energy_weight)
            _apply_reward_weight("action_rate_l2", stage_cfg.action_rate_l2_weight)
            _apply_reward_weight("joint_deviation_hip", stage_cfg.joint_deviation_hip_weight)
            _apply_reward_weight("joint_deviation_arms", stage_cfg.joint_deviation_arms_weight)
            _apply_reward_weight("joint_deviation_legs", stage_cfg.joint_deviation_legs_weight)

        stage_cfg, stage_idx, stage_max_updates, stage_progress = _resolve_stage(
            self._episode_len_curriculum_stage_successes
        )
        if stage_idx != self._episode_len_curriculum_stage_idx:
            self._episode_len_curriculum_stage_idx = stage_idx
            _apply_stage_overrides(stage_cfg)

        def _stage_value(name, default):
            if stage_cfg is None:
                return default
            value = getattr(stage_cfg, name)
            return default if value is None else value

        round_completed = False
        round_success = self._episode_len_curriculum_last_round_success
        round_reward_success = self._episode_len_curriculum_last_round_reward_success
        round_timeout_success = self._episode_len_curriculum_last_round_timeout_success
        required_streak_rounds = max(1, int(_stage_value("required_streak_rounds", cfg.required_streak_rounds)))
        required_timeout_ratio = float(_stage_value("episode_length_ratio", cfg.episode_length_ratio))
        required_timeout_ratio = float(np.clip(required_timeout_ratio, 0.0, 1.0))
        min_mean_reward = float(_stage_value("min_mean_reward", cfg.min_mean_reward))
        round_episode_count = int(_stage_value("round_episode_count", cfg.round_episode_count))
        target_episode_count = round_episode_count if round_episode_count > 0 else self.num_envs

        episode_lengths = self.episode_length_buf[env_ids].float()
        valid_mask = episode_lengths > 0.0
        # Ignore episodes that used out-of-bounds teleport so curriculum mean episode length is not biased.
        valid_mask &= ~self._episode_had_teleport_buf[env_ids]
        valid_episode_lengths = episode_lengths[valid_mask]
        episode_rewards = self.episode_reward_buf[env_ids].float()
        valid_episode_rewards = episode_rewards[valid_mask]
        timeout_episode_count = int(self.time_out_buf[env_ids][valid_mask].sum().item())
        if valid_episode_lengths.numel() > 0:
            self._episode_len_curriculum_sum += float(valid_episode_lengths.sum().item())
            self._episode_len_curriculum_count += int(valid_episode_lengths.numel())
            self._episode_reward_curriculum_sum += float(valid_episode_rewards.sum().item())
            self._episode_len_curriculum_timeout_count += timeout_episode_count

            if self._episode_len_curriculum_count >= target_episode_count:
                round_completed = True
                self._episode_len_curriculum_round += 1
                self._episode_len_curriculum_last_round_mean = self._episode_len_curriculum_sum / max(
                    self._episode_len_curriculum_count, 1
                )
                self._episode_reward_curriculum_last_round_mean = self._episode_reward_curriculum_sum / max(
                    self._episode_len_curriculum_count, 1
                )
                self._episode_len_curriculum_last_round_timeout_ratio = self._episode_len_curriculum_timeout_count / max(
                    self._episode_len_curriculum_count, 1
                )
                round_reward_success = self._episode_reward_curriculum_last_round_mean >= min_mean_reward
                round_timeout_success = self._episode_len_curriculum_last_round_timeout_ratio >= required_timeout_ratio
                round_success = round_reward_success and round_timeout_success
                self._episode_len_curriculum_last_round_reward_success = round_reward_success
                self._episode_len_curriculum_last_round_timeout_success = round_timeout_success
                self._episode_len_curriculum_last_round_success = round_success
                self._episode_len_curriculum_streak = self._episode_len_curriculum_streak + 1 if round_success else 0
                self._episode_len_curriculum_sum = 0.0
                self._episode_len_curriculum_count = 0
                self._episode_len_curriculum_timeout_count = 0
                self._episode_reward_curriculum_sum = 0.0

                if self._episode_len_curriculum_streak >= required_streak_rounds:
                    self._episode_len_curriculum_stage_successes += 1
                    self._episode_len_curriculum_speed_updates += 1
                    self._episode_len_curriculum_streak = 0
                    if getattr(cfg, "stages", None):
                        (
                            new_stage_cfg,
                            new_stage_idx,
                            new_stage_max_updates,
                            new_stage_progress,
                        ) = _resolve_stage(self._episode_len_curriculum_stage_successes)
                        if new_stage_idx != self._episode_len_curriculum_stage_idx:
                            self._episode_len_curriculum_stage_idx = new_stage_idx
                            stage_cfg = new_stage_cfg
                            stage_idx = new_stage_idx
                            stage_max_updates = new_stage_max_updates
                            stage_progress = new_stage_progress
                            _apply_stage_overrides(stage_cfg)

        x_min, x_max = self.command_generator.cfg.ranges.lin_vel_x
        self.extras["log"].update(
            {
                "Curriculum/round_index": float(self._episode_len_curriculum_round),
                "Curriculum/round_mean_episode_length": float(self._episode_len_curriculum_last_round_mean),
                "Curriculum/round_mean_episode_reward": float(self._episode_reward_curriculum_last_round_mean),
                "Curriculum/round_timeout_ratio": float(self._episode_len_curriculum_last_round_timeout_ratio),
                "Curriculum/episode_length_streak": float(self._episode_len_curriculum_streak),
                "Curriculum/required_streak_rounds": float(required_streak_rounds),
                "Curriculum/required_timeout_ratio": float(required_timeout_ratio),
                "Curriculum/min_mean_reward_target": float(min_mean_reward),
                "Curriculum/round_completed": float(round_completed),
                "Curriculum/round_success": float(round_success),
                "Curriculum/round_reward_success": float(round_reward_success),
                "Curriculum/round_timeout_success": float(round_timeout_success),
                "Curriculum/forward_speed_min": float(x_min),
                "Curriculum/forward_speed_max": float(x_max),
                "Curriculum/stage_successes": float(self._episode_len_curriculum_stage_successes),
                "Curriculum/speed_updates": float(self._episode_len_curriculum_speed_updates),
                "Curriculum/termination_contact_enabled": float(self._termination_contact_enabled),
                "Curriculum/termination_contact_delay_s": float(self._termination_contact_delay_s),
            }
        )
        if hasattr(self.cfg.reward, "track_lin_vel_xy_exp"):
            self.extras["log"]["Curriculum/track_lin_vel_xy_exp_weight"] = float(
                self.cfg.reward.track_lin_vel_xy_exp.weight
            )
        if hasattr(self.cfg.reward, "track_ang_vel_z_exp"):
            self.extras["log"]["Curriculum/track_ang_vel_z_exp_weight"] = float(
                self.cfg.reward.track_ang_vel_z_exp.weight
            )
        if stage_cfg is not None:
            self.extras["log"].update(
                {
                    "Curriculum/stage_index": float(stage_idx),
                    "Curriculum/stage_max_updates": float(stage_max_updates),
                    "Curriculum/stage_progress": float(stage_progress),
                }
            )

        if cfg.print_status and round_completed:
            step = self.sim_step_counter // self.cfg.sim.decimation
            print(
                "[CURRICULUM][LeggedLab] "
                f"step={step} round={self._episode_len_curriculum_round} "
                f"mean_ep_len={self._episode_len_curriculum_last_round_mean:.2f} "
                f"mean_ep_rew={self._episode_reward_curriculum_last_round_mean:.2f} "
                f"timeout_ratio={self._episode_len_curriculum_last_round_timeout_ratio:.3f} "
                f"success={int(round_success)} "
                f"streak={self._episode_len_curriculum_streak}/{required_streak_rounds} "
                f"lin_vel_x=({float(x_min):.3f}, {float(x_max):.3f}) "
                f"stage={stage_idx}"
            )

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        command = self._command_tensor()
        move_down = distance < torch.norm(command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        if self._rsl_rl_uses_tensordict_obs:
            return self._obs_tensor_dict(actor_obs, critic_obs)
        return actor_obs, self.extras

    def _obs_tensor_dict(self, actor_obs, critic_obs):
        if TensorDict is None:
            raise RuntimeError("This rsl_rl version requires TensorDict observations, but tensordict is not installed.")
        return TensorDict(
            {"actor": actor_obs, "critic": critic_obs},
            batch_size=[self.num_envs],
            device=self.device,
        )

    @staticmethod
    def _detect_rsl_rl_tensordict_observations() -> bool:
        try:
            annotation = VecEnv.get_observations.__annotations__.get("return", "")
        except AttributeError:
            return False
        return "TensorDict" in str(annotation)

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
