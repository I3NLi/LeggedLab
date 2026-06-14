# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.

from __future__ import annotations

from typing import Literal

import torch
import warp as wp
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op, _validate_scale_range
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


def _as_torch_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return wp.to_torch(value)


def randomize_rigid_body_com_fixed(env, env_ids: torch.Tensor | None, com_range: dict[str, tuple[float, float]], asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.int32)
    else:
        env_ids = env_ids.to(asset.device, dtype=torch.int32)

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, device=asset.device, dtype=torch.long)
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, device=asset.device, dtype=torch.long)

    ranges = torch.tensor(
        [com_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")],
        device=asset.device,
        dtype=torch.float32,
    )
    offsets = torch.empty((env_ids.numel(), 1, 3), device=asset.device, dtype=torch.float32)
    offsets.uniform_(0.0, 1.0)
    offsets = ranges[:, 0] + offsets * (ranges[:, 1] - ranges[:, 0])

    coms = wp.to_torch(asset.root_view.get_coms()).clone().to(device=asset.device, dtype=torch.float32).contiguous()
    coms[env_ids[:, None], body_ids, :3] += offsets
    coms_wp = wp.from_torch(coms.cpu().contiguous(), dtype=wp.float32)
    env_ids_wp = wp.from_torch(env_ids.cpu().contiguous(), dtype=wp.uint32)
    asset.root_view.set_coms(coms_wp, env_ids_wp)


class randomize_actuator_gains_fixed(ManagerTermBase):
    """Randomize implicit actuator PD gains and write the randomized values back to sim."""

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.default_joint_stiffness = _as_torch_tensor(self.asset.data.joint_stiffness).clone()
        self.default_joint_damping = _as_torch_tensor(self.asset.data.joint_damping).clone()

        if cfg.params["operation"] == "scale":
            if "stiffness_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["damping_distribution_params"], "damping_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_actuator_gains_fixed' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        for actuator in self.asset.actuators.values():
            resolved = self._resolve_actuator_joint_ids(actuator)
            if resolved is None:
                continue
            actuator_indices, global_indices = resolved

            if stiffness_distribution_params is not None:
                stiffness = actuator.stiffness[env_ids].clone()
                stiffness[:, actuator_indices] = self._randomize_from_default(
                    self.default_joint_stiffness, env_ids, actuator_indices, global_indices,
                    stiffness_distribution_params, operation, distribution
                )
                actuator.stiffness[env_ids] = stiffness
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_stiffness_to_sim(
                        stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids
                    )

            if damping_distribution_params is not None:
                damping = actuator.damping[env_ids].clone()
                damping[:, actuator_indices] = self._randomize_from_default(
                    self.default_joint_damping, env_ids, actuator_indices, global_indices,
                    damping_distribution_params, operation, distribution
                )
                actuator.damping[env_ids] = damping
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)

    def _resolve_actuator_joint_ids(self, actuator):
        if isinstance(self.asset_cfg.joint_ids, slice):
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            elif isinstance(actuator.joint_indices, torch.Tensor):
                global_indices = actuator.joint_indices.to(self.asset.device)
            else:
                raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
            return actuator_indices, global_indices

        if isinstance(actuator.joint_indices, slice):
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
            return joint_ids, joint_ids

        actuator_joint_indices = actuator.joint_indices
        asset_joint_ids = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
        actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
        if len(actuator_indices) == 0:
            return None
        global_indices = actuator_joint_indices[actuator_indices]
        return actuator_indices, global_indices

    def _randomize_from_default(
        self,
        default_values: torch.Tensor,
        env_ids: torch.Tensor,
        actuator_indices,
        global_indices,
        params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"],
    ) -> torch.Tensor:
        values = default_values[env_ids][:, global_indices].clone()
        return _randomize_prop_by_op(
            values,
            params,
            dim_0_ids=None,
            dim_1_ids=slice(None),
            operation=operation,
            distribution=distribution,
        )
