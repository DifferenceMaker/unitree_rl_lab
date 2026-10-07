"""Domain-randomization pieces the trunks never had (2026-10-07, the p16dr wave).

Every balance / walk trunk has run with DR effectively OFF since p6 (friction fixed 1.0, base mass +0, PD-gain scale 1.0; only the
0-3 kg hand payload and the pushes). The one sim2real symptom nothing in that list can represent is an ENCODER ZERO OFFSET: the
real robot reports a joint angle that differs from the true angle by a constant (the pre-recalibration ankle-roll pair carried a
4.4-4.7 deg sum offset), so a policy that trusts "0 rad = vertical" stands on a rolled foot and compensates up the chain.

`randomize_encoder_offset` (event, reset mode): every episode each selected joint draws its own constant offset. The policy observes
joint_pos_rel + offset (`joint_pos_rel_encoder`) and its position target is applied MINUS the offset (`RobustJointPositionAction`),
which is exactly what a real encoder error does to a PD loop (the PD runs in the encoder frame; the physical joint lands at
target - offset). The critic keeps the clean joint_pos_rel (privileged, same dimension as before -> warmstart-compatible). Over
training "zero rad means vertical" stops being true and uprightness has to come from projected gravity, base angular velocity and foot
contact -- the mechanism a model-based controller with a state estimator has for free.

`RobustJointPositionAction` also carries an optional per-env action LAG of 0..max_lag_steps POLICY steps (20 ms each at 50 Hz),
drawn per reset: lag 1 applies the previous step's target. Finer (physics-step) delays would need the DelayedPDActuator = a plant
change; this stays inside the action term so the actuator model is untouched. With max_lag_steps 0 and no encoder event the term is
bit-identical to JointPositionAction (the trunk default).
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


def encoder_offset_buffer(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """The per-env, per-joint encoder offset tensor (num_envs x num_joints), zeros until an event fills it."""
    buf = getattr(env, "encoder_offset", None)
    if buf is None:
        asset: Articulation = env.scene[asset_name]
        buf = torch.zeros(env.num_envs, asset.num_joints, device=env.device)
        env.encoder_offset = buf
    return buf


def randomize_encoder_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    offset_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset-mode event: draw a constant per-joint encoder offset (rad) for each env in env_ids, uniform in offset_range,
    for the joints in asset_cfg (the others stay 0)."""
    buf = encoder_offset_buffer(env, asset_cfg.name)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    lo, hi = float(offset_range[0]), float(offset_range[1])
    samples = torch.empty(len(env_ids), buf.shape[1], device=env.device).uniform_(lo, hi)
    if asset_cfg.joint_ids != slice(None):
        mask = torch.zeros(buf.shape[1], dtype=torch.bool, device=env.device)
        mask[torch.as_tensor(asset_cfg.joint_ids, device=env.device)] = True
        samples = samples * mask
    buf[env_ids] = samples


def joint_pos_rel_encoder(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint_pos_rel as the ENCODER reports it: true relative position + the env's constant offset."""
    asset: Articulation = env.scene[asset_cfg.name]
    buf = encoder_offset_buffer(env, asset_cfg.name)
    return (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]) + buf[:, asset_cfg.joint_ids]


def encoder_offset_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Privileged: the offsets themselves (NOT used by the p16dr critic -- it would change the critic width)."""
    return encoder_offset_buffer(env, asset_cfg.name)[:, asset_cfg.joint_ids]


class RobustJointPositionAction(JointPositionAction):
    """JointPositionAction whose applied target is (raw * scale + default) - encoder_offset, optionally lagged 0..N policy steps."""

    cfg: "RobustJointPositionActionCfg"

    def __init__(self, cfg: "RobustJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._lag = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_target = torch.zeros_like(self._processed_actions)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        enc = getattr(self._env, "encoder_offset", None)
        if enc is not None:
            self._processed_actions = self._processed_actions - enc[:, self._joint_ids]
        if self.cfg.max_lag_steps > 0:
            target = self._processed_actions
            use_prev = (self._lag > 0) & ~self._fresh
            self._processed_actions = torch.where(use_prev.unsqueeze(1), self._prev_target, target)
            self._prev_target = target.clone()
            self._fresh[:] = False

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        n = self.num_envs if env_ids == slice(None) else len(env_ids)
        if self.cfg.max_lag_steps > 0:
            self._lag[env_ids] = torch.randint(0, self.cfg.max_lag_steps + 1, (n,), device=self.device)
        self._fresh[env_ids] = True


@configclass
class RobustJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = RobustJointPositionAction
    max_lag_steps: int = 0
    """Per-env action lag drawn uniformly in [0, max_lag_steps] POLICY steps at every reset (0 = off)."""
