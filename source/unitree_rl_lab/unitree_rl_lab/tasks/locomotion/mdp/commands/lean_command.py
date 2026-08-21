"""dp6 lean command (colleague's design, operator-ratified 2026-08-21).

A 1-dim commanded pelvis/torso PITCH in RADIANS (gravity-referenced,
forward-positive). Training samples it; at deploy the colleague's publisher
feeds the same obs slot. Range [-0.10, +0.35] (-6..+20 deg — forward lean is
the use case); zero_prob resamples snap to exactly 0 so normal upright work
stays trained and UNCOMMANDED lean is explicitly priced by the tracking pair
(lean_track_bonus / flat_orientation_lean_l2).
"""
from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import CommandTermCfg
from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass


class LeanCommand(CommandTerm):
    """Uniform 1-dim pitch command with a zero-snap fraction."""

    cfg: "LeanCommandCfg"

    def __init__(self, cfg: "LeanCommandCfg", env):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 1, device=self.device)

    def __str__(self) -> str:
        return f"LeanCommand(range={self.cfg.ranges}, zero_prob={self.cfg.zero_prob})"

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids):
        lo, hi = self.cfg.ranges
        r = torch.rand(len(env_ids), 1, device=self.device)
        cmd = lo + (hi - lo) * r
        zero = torch.rand(len(env_ids), 1, device=self.device) < self.cfg.zero_prob
        self._command[env_ids] = torch.where(zero, torch.zeros_like(cmd), cmd)

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass


@configclass
class LeanCommandCfg(CommandTermCfg):
    class_type: type = LeanCommand
    ranges: tuple = MISSING           # (lo, hi) rad, forward-positive
    zero_prob: float = 0.3            # fraction of resamples snapped to 0
    resampling_time_range: tuple = (4.0, 8.0)
