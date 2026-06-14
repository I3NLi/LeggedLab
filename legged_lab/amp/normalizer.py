# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.

from __future__ import annotations

import torch


class RunningNormalizer:
    """Running mean/std normalizer for AMP observations."""

    def __init__(self, dim: int, device: str, eps: float = 1.0e-5) -> None:
        self.dim = int(dim)
        self.device = device
        self.eps = float(eps)
        self.mean = torch.zeros(self.dim, device=device)
        self.var = torch.ones(self.dim, device=device)
        self.count = torch.tensor(self.eps, device=device)

    @torch.no_grad()
    def update(self, samples: torch.Tensor) -> None:
        if samples.numel() == 0:
            return
        flat = samples.detach().reshape(-1, self.dim).to(self.device)
        batch_count = torch.tensor(float(flat.shape[0]), device=self.device)
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.square() * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = (m_2 / total_count).clamp_min(self.eps)
        self.count = total_count

    def normalize(self, samples: torch.Tensor) -> torch.Tensor:
        return (samples - self.mean) / torch.sqrt(self.var + self.eps)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.mean = state_dict["mean"].to(self.device)
        self.var = state_dict["var"].to(self.device)
        self.count = state_dict["count"].to(self.device)
