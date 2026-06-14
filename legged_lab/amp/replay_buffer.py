# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.

from __future__ import annotations

import torch


class AMPReplayBuffer:
    """Fixed-size circular buffer for policy AMP observation windows."""

    def __init__(self, observation_dim: int, buffer_size: int, num_frames: int, device: str) -> None:
        if buffer_size <= 0:
            raise ValueError(f"AMP replay buffer size must be positive, got {buffer_size}.")
        self.device = device
        self.buffer_size = int(buffer_size)
        self.num_frames = int(num_frames)
        self.observation_dim = int(observation_dim)
        self.states = torch.zeros(
            self.buffer_size,
            self.num_frames,
            self.observation_dim,
            device=device,
        )
        self.step = 0
        self.num_samples = 0

    @torch.no_grad()
    def insert(self, states: torch.Tensor) -> None:
        states = states.detach().to(self.device)
        if states.ndim != 3 or states.shape[1:] != (self.num_frames, self.observation_dim):
            raise ValueError(
                "AMP replay states must have shape "
                f"(N, {self.num_frames}, {self.observation_dim}), got {tuple(states.shape)}."
            )

        num_states = int(states.shape[0])
        if num_states >= self.buffer_size:
            self.states[:] = states[-self.buffer_size :]
            self.step = 0
            self.num_samples = self.buffer_size
            return

        end = self.step + num_states
        if end <= self.buffer_size:
            self.states[self.step : end] = states
        else:
            first = self.buffer_size - self.step
            self.states[self.step :] = states[:first]
            self.states[: end - self.buffer_size] = states[first:]

        self.step = end % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + num_states)

    def mini_batch_generator(self, num_batches: int, mini_batch_size: int):
        if self.num_samples <= 0:
            raise RuntimeError("AMP replay buffer is empty.")
        for _ in range(num_batches):
            indices = torch.randint(self.num_samples, (mini_batch_size,), device=self.device)
            yield self.states[indices]
