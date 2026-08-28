from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class CornerVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand + CORNER SAMPLING (lm5e, operator 2026-08-28: "train
    multiple commanded velocities at once … 20-30 % of the time").

    Commands were always sampled jointly (uniform on all three axes per resample);
    what stays rare is the CORNERS — large magnitudes on several axes at once —
    because the per-axis curriculum widens each range on its own tracking score.
    With probability `corner_prob` a resample snaps every axis to an edge of its
    CURRENT range (random sign per axis), so back+left, fast+turn, … appear at
    full magnitude. Edges follow the curriculum (cfg.ranges is mutated by the
    level terms). Standing envs are still zeroed by _update_command."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        p = float(getattr(self.cfg, "corner_prob", 0.0) or 0.0)
        if p <= 0.0 or len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device)
        pick = torch.rand(len(ids), device=self.device) < p
        if not pick.any():
            return
        sel = ids[pick]
        for axis, rng in enumerate((self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y, self.cfg.ranges.ang_vel_z)):
            hi_side = torch.rand(len(sel), device=self.device) < 0.5
            self.vel_command_b[sel, axis] = torch.where(
                hi_side, torch.full_like(hi_side, rng[1], dtype=torch.float),
                torch.full_like(hi_side, rng[0], dtype=torch.float))


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = CornerVelocityCommand
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    corner_prob: float = 0.0
    """Fraction of resamples snapped to the range corners (0 = plain uniform, the
    behaviour of every run before lm5e)."""
