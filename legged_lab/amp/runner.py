# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.

from __future__ import annotations

import os
import statistics
import time
import torch
import warnings
from collections import deque

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, store_code_state

from legged_lab.amp.ppo import AMPPPO


class AMPOnPolicyRunner(OnPolicyRunner):
    """RSL-RL runner with adversarial motion prior rollout and update hooks."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.motion_prior_cfg = dict(train_cfg.get("motion_prior", {}))
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.git_status_repos.append(__file__)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:  # noqa: C901
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        amp_obs = self.env.get_amp_observations().to(self.device)
        amp_obs_frames = amp_obs.unsqueeze(1).repeat(1, self.alg.amp_num_frames, 1)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        amp_rewbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_amp_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            amp_step_reward_sum = 0.0
            amp_step_logit_sum = 0.0
            amp_step_gate_sum = 0.0
            amp_step_replay_gate_sum = 0.0
            amp_step_count = 0

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))

                    next_amp_obs = self.env.get_amp_observations().to(self.device)
                    amp_obs_frames = torch.cat((amp_obs_frames[:, 1:], next_amp_obs.unsqueeze(1)), dim=1)
                    command_values = self._command_values()
                    command_speeds = (
                        torch.norm(command_values[:, :2].to(self.device), dim=-1)
                        if command_values is not None
                        else None
                    )
                    rewards_with_amp, amp_rewards, amp_logits, amp_gate = self.alg.predict_amp_reward(
                        amp_obs_frames,
                        rewards,
                        command_speeds=command_speeds,
                        command_values=command_values,
                    )
                    amp_replay_gate = self.alg.amp_reward_gate(command_speeds=None, command_values=command_values)

                    self.alg.process_env_step(
                        obs,
                        rewards_with_amp,
                        dones,
                        extras,
                        amp_obs_frames=amp_obs_frames,
                        amp_replay_gate=amp_replay_gate,
                    )

                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    if done_ids.numel() > 0:
                        amp_obs_frames[done_ids] = next_amp_obs[done_ids].unsqueeze(1).repeat(
                            1, self.alg.amp_num_frames, 1
                        )

                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    amp_step_reward_sum += float(amp_rewards.mean().item())
                    amp_step_logit_sum += float(amp_logits.mean().item())
                    amp_step_gate_sum += float(amp_gate.mean().item())
                    amp_step_replay_gate_sum += float(amp_replay_gate.mean().item())
                    amp_step_count += 1

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                        if self.alg.rnd:
                            cur_ereward_sum += rewards_with_amp
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards_with_amp + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards_with_amp
                        cur_amp_reward_sum += amp_rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        amp_rewbuffer.extend(cur_amp_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        cur_amp_reward_sum[new_ids] = 0

                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            mean_amp_step_reward = amp_step_reward_sum / max(amp_step_count, 1)
            mean_amp_step_logit = amp_step_logit_sum / max(amp_step_count, 1)
            mean_amp_step_gate = amp_step_gate_sum / max(amp_step_count, 1)
            mean_amp_step_replay_gate = amp_step_replay_gate_sum / max(amp_step_count, 1)

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width=width, pad=pad)
        if self.writer is None or self.disable_logs:
            return
        if len(locs["amp_rewbuffer"]) > 0:
            self.writer.add_scalar("AMP/mean_episode_reward", statistics.mean(locs["amp_rewbuffer"]), locs["it"])
        self.writer.add_scalar("AMP/mean_step_reward", locs["mean_amp_step_reward"], locs["it"])
        self.writer.add_scalar("AMP/mean_step_logit", locs["mean_amp_step_logit"], locs["it"])
        self.writer.add_scalar("AMP/mean_step_gate", locs["mean_amp_step_gate"], locs["it"])
        self.writer.add_scalar("AMP/mean_step_replay_gate", locs["mean_amp_step_replay_gate"], locs["it"])

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "amp_discriminator_state_dict": self.alg.discriminator.state_dict(),
            "amp_discriminator_optimizer_state_dict": self.alg.discriminator_optimizer.state_dict(),
            "amp_normalizer_state_dict": self.alg.amp_normalizer.state_dict(),
        }
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if "amp_discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"])
        if "amp_normalizer_state_dict" in loaded_dict:
            self.alg.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"])
        if hasattr(self.alg, "rnd") and self.alg.rnd and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

        if load_optimizer and resumed_training:
            if "optimizer_state_dict" in loaded_dict:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "amp_discriminator_optimizer_state_dict" in loaded_dict:
                self.alg.discriminator_optimizer.load_state_dict(
                    loaded_dict["amp_discriminator_optimizer_state_dict"]
                )
            if hasattr(self.alg, "rnd") and self.alg.rnd and "rnd_optimizer_state_dict" in loaded_dict:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def train_mode(self) -> None:
        super().train_mode()
        self.alg.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.alg.discriminator.eval()

    def _construct_algorithm(self, obs) -> AMPPPO:
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        if not hasattr(self.env, "get_amp_observations") or not hasattr(self.env, "collect_reference_amp_observations"):
            raise RuntimeError("AMP runner requires env.get_amp_observations() and env.collect_reference_amp_observations().")

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        alg_cfg = dict(self.alg_cfg)
        alg_cfg.pop("class_name", None)
        motion_prior_cfg = self.motion_prior_cfg
        default_num_frames = getattr(self.env.reference_motion, "amp_observation_history_length", 2)
        num_frames = int(motion_prior_cfg.get("num_frames", default_num_frames))
        amp_obs_dim = int(self.env.get_amp_observations().shape[-1])

        alg = AMPPPO(
            actor_critic,
            expert_sampler=self.env.collect_reference_amp_observations,
            amp_observation_dim=amp_obs_dim,
            amp_num_frames=num_frames,
            amp_replay_buffer_size=int(motion_prior_cfg.get("replay_buffer_size", 100_000)),
            amp_reward_coef=float(motion_prior_cfg.get("reward_coef", 0.2)),
            amp_reward_min_command_speed=float(motion_prior_cfg.get("reward_min_command_speed", 2.0)),
            amp_reward_max_command_y_abs=float(motion_prior_cfg.get("reward_max_command_y_abs", 0.0)),
            amp_reward_command_y_gate_width=float(motion_prior_cfg.get("reward_command_y_gate_width", 0.0)),
            amp_reward_max_command_yaw_abs=float(motion_prior_cfg.get("reward_max_command_yaw_abs", 0.0)),
            amp_reward_command_yaw_gate_width=float(motion_prior_cfg.get("reward_command_yaw_gate_width", 0.0)),
            amp_discriminator_hidden_dims=list(motion_prior_cfg.get("discriminator_hidden_dims", [256, 128])),
            amp_discriminator_learning_rate=float(motion_prior_cfg.get("discriminator_learning_rate", 1.0e-4)),
            amp_discriminator_weight_decay=float(motion_prior_cfg.get("discriminator_weight_decay", 1.0e-4)),
            amp_grad_penalty_coef=float(motion_prior_cfg.get("grad_penalty_coef", 5.0)),
            amp_use_spectral_norm=bool(motion_prior_cfg.get("use_spectral_norm", True)),
            device=self.device,
            **alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return alg

    def _command_values(self) -> torch.Tensor | None:
        if hasattr(self.env, "_command_tensor"):
            return self.env._command_tensor()
        elif hasattr(self.env, "command_generator") and hasattr(self.env.command_generator, "vel_command_b"):
            return self.env.command_generator.vel_command_b
        return None

    def _command_speeds(self) -> torch.Tensor | None:
        command = self._command_values()
        if command is None:
            return None
        return torch.norm(command[:, :2].to(self.device), dim=-1)
