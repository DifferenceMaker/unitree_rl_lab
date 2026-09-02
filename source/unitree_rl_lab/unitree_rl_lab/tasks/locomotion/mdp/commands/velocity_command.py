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
        if len(env_ids) == 0:
            return
        p_corner = float(getattr(self.cfg, "corner_prob", 0.0) or 0.0)
        p_slow = float(getattr(self.cfg, "slow_prob", 0.0) or 0.0)
        p_spin = float(getattr(self.cfg, "spin_prob", 0.0) or 0.0)
        if p_corner + p_slow + p_spin <= 0.0:
            return
        ids = torch.as_tensor(env_ids, device=self.device)
        r = torch.rand(len(ids), device=self.device)
        # exclusive buckets from one draw: [0,p_corner) corner, then slow, then spin
        corner = r < p_corner
        slow = (~corner) & (r < p_corner + p_slow)
        spin = (~corner) & (~slow) & (r < p_corner + p_slow + p_spin)
        if corner.any():
            sel = ids[corner]
            for axis, rng in enumerate((self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y, self.cfg.ranges.ang_vel_z)):
                hi_side = torch.rand(len(sel), device=self.device) < 0.5
                self.vel_command_b[sel, axis] = torch.where(
                    hi_side, torch.full_like(hi_side, rng[1], dtype=torch.float),
                    torch.full_like(hi_side, rng[0], dtype=torch.float))
        # lm5g SLOW band (operator 2026-09-02: sub-0.2 striding is the weak zone in
        # every lm5f run; "beautiful gait at small speeds ... transfer to desk line"):
        # resample lin vel into +-slow_band, wz kept as drawn.
        if slow.any():
            sel = ids[slow]
            band = float(getattr(self.cfg, "slow_band", 0.3) or 0.3)
            for axis, rng in ((0, self.cfg.ranges.lin_vel_x), (1, self.cfg.ranges.lin_vel_y)):
                lo = max(rng[0], -band)
                hi = min(rng[1], band)
                self.vel_command_b[sel, axis] = torch.empty(len(sel), device=self.device).uniform_(lo, hi)
        # lm5g SPIN (operator: "still no rotation while standing" on 2 of 3 lm5f runs;
        # "make it worth rotating"): pure rotation-in-place — vx=vy=0, wz drawn AWAY
        # from zero (sign * U(spin_min, range hi)) so the sample is a real rotation
        # demand, not a dead zone draw. Standing envs are still zeroed downstream.
        if spin.any():
            sel = ids[spin]
            self.vel_command_b[sel, 0] = 0.0
            self.vel_command_b[sel, 1] = 0.0
            wlo, whi = self.cfg.ranges.ang_vel_z
            mag_hi = max(abs(wlo), abs(whi))
            smin = min(float(getattr(self.cfg, "spin_min", 0.2) or 0.2), mag_hi)
            mag = torch.empty(len(sel), device=self.device).uniform_(smin, mag_hi)
            sign = torch.where(torch.rand(len(sel), device=self.device) < 0.5, -1.0, 1.0)
            self.vel_command_b[sel, 2] = mag * sign


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = CornerVelocityCommand
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    corner_prob: float = 0.0
    """Fraction of resamples snapped to the range corners (0 = plain uniform, the
    behaviour of every run before lm5e)."""
    slow_prob: float = 0.0
    """lm5g: fraction of resamples with lin vel redrawn inside +-slow_band (wz kept)."""
    slow_band: float = 0.3
    spin_prob: float = 0.0
    """lm5g: fraction of resamples set to rotation-in-place (vx=vy=0, |wz| >= spin_min)."""
    spin_min: float = 0.2
