from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
import torch
from isaaclab.utils import configclass


@configclass
class CurriculumCfg:
    enable: bool = False
    mean_reward_thresholds: tuple[float, ...] = ()
    avg_window: int = 100
    min_window: int = 100
    speed_curriculum_enable: bool = False
    lin_vel_x_min_start: float | None = None
    lin_vel_x_min_step: float = 0.0
    lin_vel_x_min_limit: float | None = None
    lin_vel_x_max_start: float | None = None
    lin_vel_x_max_step: float = 0.0
    lin_vel_x_max_limit: float | None = None
    lin_vel_y_max_start: float | None = None
    lin_vel_y_max_step: float = 0.0
    lin_vel_y_max_limit: float | None = None
    ang_vel_z_max_start: float | None = None
    ang_vel_z_max_step: float = 0.0
    ang_vel_z_max_limit: float | None = None


class CurriculumManager:
    def __init__(self, cfg: CurriculumCfg, on_stage_change: Callable[[int], None] | None = None):
        self.cfg = cfg
        self._enabled = bool(cfg.enable and len(cfg.mean_reward_thresholds) > 0)
        self._thresholds = list(cfg.mean_reward_thresholds)
        self._next_idx = 0
        self.stage = 0
        self.last_mean: float | None = None
        self._reward_window = deque(maxlen=cfg.avg_window)
        self._on_stage_change = on_stage_change
        self._auto_step: float | None = None
        self._next_threshold: float | None = self._thresholds[0] if self._thresholds else None

    def update(self, returns: torch.Tensor, log_dict: dict | None = None) -> None:
        if not self._enabled or returns.numel() == 0:
            return

        self._reward_window.extend(returns.detach().cpu().tolist())
        if len(self._reward_window) < self.cfg.min_window:
            return

        mean_reward = float(np.mean(self._reward_window))
        self.last_mean = mean_reward
        if self._next_idx < len(self._thresholds):
            next_threshold = self._thresholds[self._next_idx]
        else:
            next_threshold = self._next_threshold

        if next_threshold is None:
            return

        if mean_reward >= next_threshold:
            self._next_idx += 1
            self.stage = self._next_idx
            if self._next_idx >= len(self._thresholds):
                if self._auto_step is None:
                    if len(self._thresholds) >= 2:
                        self._auto_step = self._thresholds[-1] - self._thresholds[-2]
                    else:
                        self._auto_step = 1.0
                    if self._auto_step <= 0:
                        self._auto_step = 1.0
                if self._next_threshold is None:
                    self._next_threshold = self._thresholds[-1] + self._auto_step
                else:
                    self._next_threshold += self._auto_step
            if log_dict is not None:
                log_dict["Curriculum/mean_reward"] = mean_reward
                log_dict["Curriculum/stage"] = self.stage
            if self._on_stage_change is not None:
                self._on_stage_change(self.stage)

    @property
    def enabled(self) -> bool:
        return self._enabled
