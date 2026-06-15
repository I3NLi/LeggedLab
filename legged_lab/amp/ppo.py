# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.utils import string_to_callable

from legged_lab.amp.discriminator import AMPDiscriminator
from legged_lab.amp.normalizer import RunningNormalizer
from legged_lab.amp.replay_buffer import AMPReplayBuffer


class AMPPPO(PPO):
    """PPO with an adversarial motion prior discriminator reward."""

    def __init__(
        self,
        policy,
        expert_sampler: Callable[[int, int], torch.Tensor],
        amp_observation_dim: int,
        amp_num_frames: int = 2,
        amp_replay_buffer_size: int = 100_000,
        amp_reward_coef: float = 0.2,
        amp_reward_min_command_speed: float = 2.0,
        amp_reward_max_command_y_abs: float = 0.0,
        amp_reward_command_y_gate_width: float = 0.0,
        amp_reward_max_command_yaw_abs: float = 0.0,
        amp_reward_command_yaw_gate_width: float = 0.0,
        amp_expert_command_conditioning: bool = False,
        amp_expert_command_dim: int = 3,
        amp_discriminator_hidden_dims: list[int] | tuple[int, ...] | None = None,
        amp_discriminator_learning_rate: float = 1.0e-4,
        amp_discriminator_weight_decay: float = 1.0e-4,
        amp_grad_penalty_coef: float = 5.0,
        amp_use_spectral_norm: bool = True,
        **ppo_kwargs,
    ) -> None:
        super().__init__(policy, **ppo_kwargs)

        hidden_dims = [256, 128] if amp_discriminator_hidden_dims is None else list(amp_discriminator_hidden_dims)
        self.amp_observation_dim = int(amp_observation_dim)
        self.amp_num_frames = int(amp_num_frames)
        self.amp_reward_coef = float(amp_reward_coef)
        self.amp_reward_min_command_speed = float(amp_reward_min_command_speed)
        self.amp_reward_max_command_y_abs = float(amp_reward_max_command_y_abs)
        self.amp_reward_command_y_gate_width = float(amp_reward_command_y_gate_width)
        self.amp_reward_max_command_yaw_abs = float(amp_reward_max_command_yaw_abs)
        self.amp_reward_command_yaw_gate_width = float(amp_reward_command_yaw_gate_width)
        self.amp_expert_command_conditioning = bool(amp_expert_command_conditioning)
        self.amp_expert_command_dim = int(amp_expert_command_dim)
        self.amp_grad_penalty_coef = float(amp_grad_penalty_coef)
        self.expert_sampler = expert_sampler

        self.discriminator = AMPDiscriminator(
            observation_dim=self.amp_observation_dim,
            num_frames=self.amp_num_frames,
            hidden_dims=hidden_dims,
            use_spectral_norm=amp_use_spectral_norm,
        ).to(self.device)
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=float(amp_discriminator_learning_rate),
            weight_decay=float(amp_discriminator_weight_decay),
        )
        self.amp_normalizer = RunningNormalizer(self.amp_observation_dim, self.device)
        self.amp_replay_buffer = AMPReplayBuffer(
            self.amp_observation_dim,
            int(amp_replay_buffer_size),
            self.amp_num_frames,
            self.device,
            command_dim=self.amp_expert_command_dim if self.amp_expert_command_conditioning else 0,
        )

    def predict_amp_reward(
        self,
        amp_obs_frames: torch.Tensor,
        task_rewards: torch.Tensor,
        command_speeds: torch.Tensor | None = None,
        command_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.discriminator.eval()
            normalized = self.amp_normalizer.normalize(amp_obs_frames.to(self.device))
            logits = self.discriminator(normalized).squeeze(-1)
            amp_rewards = self.amp_reward_coef * torch.clamp(1.0 - 0.25 * (logits - 1.0).square(), min=0.0)
            reward_gate = self.amp_reward_gate(command_speeds=command_speeds, command_values=command_values)
            amp_rewards *= reward_gate
            self.discriminator.train()
        return task_rewards + amp_rewards.view_as(task_rewards), amp_rewards, logits, reward_gate

    def process_env_step(
        self,
        obs,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
        amp_obs_frames: torch.Tensor,
        amp_replay_gate: torch.Tensor | None = None,
        command_values: torch.Tensor | None = None,
    ) -> None:
        commands = self._prepare_amp_commands(command_values, num_rows=int(amp_obs_frames.shape[0]))
        if amp_replay_gate is None:
            self.amp_replay_buffer.insert(amp_obs_frames, commands=commands)
        else:
            keep_ids = (amp_replay_gate.to(self.device) > 0.0).nonzero(as_tuple=False).flatten()
            if keep_ids.numel() > 0:
                kept_commands = commands[keep_ids] if commands is not None else None
                self.amp_replay_buffer.insert(amp_obs_frames[keep_ids], commands=kept_commands)
        super().process_env_step(obs, rewards, dones, extras)

    def amp_reward_gate(
        self,
        command_speeds: torch.Tensor | None = None,
        command_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if command_speeds is not None:
            gate = torch.ones_like(command_speeds, device=self.device)
            if self.amp_reward_min_command_speed > 0.0:
                gate *= (command_speeds.to(self.device) >= self.amp_reward_min_command_speed).float()
        elif command_values is not None:
            gate = torch.ones(command_values.shape[0], device=self.device)
        else:
            return torch.ones(1, device=self.device)

        if command_values is None:
            return gate
        commands = command_values.to(self.device)
        if self.amp_reward_max_command_y_abs > 0.0:
            gate *= self._upper_abs_command_gate(
                torch.abs(commands[:, 1]),
                self.amp_reward_max_command_y_abs,
                self.amp_reward_command_y_gate_width,
            )
        if self.amp_reward_max_command_yaw_abs > 0.0:
            gate *= self._upper_abs_command_gate(
                torch.abs(commands[:, 2]),
                self.amp_reward_max_command_yaw_abs,
                self.amp_reward_command_yaw_gate_width,
            )
        return gate

    @staticmethod
    def _upper_abs_command_gate(command_abs: torch.Tensor, max_abs: float, gate_width: float) -> torch.Tensor:
        if gate_width <= 0.0:
            return (command_abs <= float(max_abs)).float()
        return torch.clamp((float(max_abs) + float(gate_width) - command_abs) / float(gate_width), 0.0, 1.0)

    def _prepare_amp_commands(self, command_values: torch.Tensor | None, num_rows: int) -> torch.Tensor | None:
        if not self.amp_expert_command_conditioning:
            return None
        if command_values is None:
            return torch.zeros(num_rows, self.amp_expert_command_dim, device=self.device)
        commands = command_values.to(self.device)
        if commands.ndim != 2 or commands.shape[1] < self.amp_expert_command_dim:
            raise ValueError(
                "AMP command conditioning expects commands with shape "
                f"(N, >= {self.amp_expert_command_dim}), got {tuple(commands.shape)}."
            )
        return commands[:, : self.amp_expert_command_dim]

    def _sample_expert_amp(self, batch_size: int, command_values: torch.Tensor | None = None) -> torch.Tensor:
        if self.amp_expert_command_conditioning and command_values is not None:
            samples = self.expert_sampler(
                batch_size,
                self.amp_num_frames,
                command_values=command_values.to(self.device),
            ).to(self.device)
        else:
            samples = self.expert_sampler(batch_size, self.amp_num_frames).to(self.device)
        expected_shape = (batch_size, self.amp_num_frames, self.amp_observation_dim)
        if tuple(samples.shape) != expected_shape:
            raise RuntimeError(f"Expert AMP samples must have shape {expected_shape}, got {tuple(samples.shape)}.")
        return samples

    def _compute_discriminator_loss(
        self, policy_amp: torch.Tensor, expert_amp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            policy_norm = self.amp_normalizer.normalize(policy_amp.detach())
            expert_norm = self.amp_normalizer.normalize(expert_amp.detach())

        policy_logits = self.discriminator(policy_norm)
        expert_logits = self.discriminator(expert_norm)
        policy_loss = nn.functional.mse_loss(policy_logits, -torch.ones_like(policy_logits))
        expert_loss = nn.functional.mse_loss(expert_logits, torch.ones_like(expert_logits))
        discriminator_loss = 0.5 * (policy_loss + expert_loss)
        grad_penalty = self.discriminator.compute_grad_penalty(expert_norm, self.amp_grad_penalty_coef)
        return discriminator_loss, grad_penalty, policy_logits, expert_logits

    def update(self) -> dict[str, float]:  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_discriminator_loss = 0.0
        mean_amp_grad_penalty = 0.0
        mean_amp_policy_logit = 0.0
        mean_amp_expert_logit = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        num_updates = self.num_learning_epochs * self.num_mini_batches
        amp_policy_generator = self.amp_replay_buffer.mini_batch_generator(
            num_updates,
            mini_batch_size,
            return_commands=self.amp_expert_command_conditioning,
        )

        for sample, policy_amp_sample in zip(generator, amp_policy_generator):
            if self.amp_expert_command_conditioning:
                policy_amp_batch, policy_command_batch = policy_amp_sample
            else:
                policy_amp_batch = policy_amp_sample
                policy_command_batch = None
            (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hidden_states_batch,
                masks_batch,
            ) = sample

            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1.0e-8
                    )

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None,
                    actions=action_mean_orig,
                    env=self.symmetry["_env"],
                )
                symmetry_loss = nn.functional.mse_loss(
                    mean_actions_batch[original_batch_size:],
                    actions_mean_symm_batch.detach()[original_batch_size:],
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = nn.functional.mse_loss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            expert_amp_batch = self._sample_expert_amp(mini_batch_size, command_values=policy_command_batch)
            discriminator_loss, grad_penalty, policy_logits, expert_logits = self._compute_discriminator_loss(
                policy_amp_batch, expert_amp_batch
            )
            discriminator_total_loss = discriminator_loss + grad_penalty
            self.discriminator_optimizer.zero_grad()
            discriminator_total_loss.backward()
            if self.is_multi_gpu:
                self.reduce_discriminator_parameters()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.discriminator_optimizer.step()
            self.amp_normalizer.update(policy_amp_batch)
            self.amp_normalizer.update(expert_amp_batch)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_amp_discriminator_loss += discriminator_loss.item()
            mean_amp_grad_penalty += grad_penalty.item()
            mean_amp_policy_logit += policy_logits.mean().item()
            mean_amp_expert_logit += expert_logits.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_amp_discriminator_loss /= num_updates
        mean_amp_grad_penalty /= num_updates
        mean_amp_policy_logit /= num_updates
        mean_amp_expert_logit /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp_discriminator": mean_amp_discriminator_loss,
            "amp_grad_penalty": mean_amp_grad_penalty,
            "amp_policy_logit": mean_amp_policy_logit,
            "amp_expert_logit": mean_amp_expert_logit,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict(), self.discriminator.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        self.discriminator.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    def reduce_discriminator_parameters(self) -> None:
        grads = [param.grad.view(-1) for param in self.discriminator.parameters() if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.discriminator.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
