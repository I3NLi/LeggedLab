# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd
from torch.nn.utils import spectral_norm


class AMPDiscriminator(nn.Module):
    """Least-squares AMP discriminator over short motion-observation windows."""

    def __init__(
        self,
        observation_dim: int,
        num_frames: int,
        hidden_dims: list[int] | tuple[int, ...],
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_frames < 1:
            raise ValueError(f"AMP num_frames must be >= 1, got {num_frames}.")
        if not hidden_dims:
            raise ValueError("AMP discriminator requires at least one hidden layer.")

        input_dim = int(observation_dim) * int(num_frames)
        layers: list[nn.Module] = []
        for hidden_dim in hidden_dims:
            linear = nn.Linear(input_dim, int(hidden_dim))
            layers.append(spectral_norm(linear) if use_spectral_norm else linear)
            layers.append(nn.ReLU())
            input_dim = int(hidden_dim)
        self.trunk = nn.Sequential(*layers)
        output = nn.Linear(input_dim, 1)
        self.output = spectral_norm(output) if use_spectral_norm else output

    def forward(self, amp_obs_frames: torch.Tensor) -> torch.Tensor:
        return self.output(self.trunk(amp_obs_frames.flatten(1)))

    def compute_grad_penalty(self, expert_obs_frames: torch.Tensor, coefficient: float) -> torch.Tensor:
        if coefficient <= 0.0:
            return torch.zeros((), device=expert_obs_frames.device)

        expert_data = expert_obs_frames.detach().clone().flatten(1)
        expert_data.requires_grad_(True)
        logits = self.output(self.trunk(expert_data))
        grad = autograd.grad(
            outputs=logits,
            inputs=expert_data,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return float(coefficient) * grad.norm(2, dim=1).square().mean()
