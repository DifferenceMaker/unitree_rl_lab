from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass

# bucket codes returned by sample_command_buckets (one per resampled env)
BUCKET_NAMES = ("uniform", "corner", "slow", "spin", "single_vx", "single_vy", "single_wz")
BUCKET_SINGLE = 4   # codes >= 4 = the single-axis bucket; code - 4 = the axis (0 vx, 1 vy, 2 wz)


def sample_command_buckets(cmd: torch.Tensor, lin_vel_x, lin_vel_y, ang_vel_z, corner_prob: float = 0.0, slow_prob: float = 0.0,
                           slow_band: float = 0.3, spin_prob: float = 0.0, spin_min: float = 0.2, single_prob: float = 0.0,
                           single_min: tuple = (0.03, 0.03, 0.05)) -> torch.Tensor | None:
    """The bucket overwrite of CornerVelocityCommand, as a pure function (no simulator): `cmd` (n, 3) = the uniform (vx, vy, wz)
    draws of the resampled envs, overwritten IN PLACE; the ranges are the CURRENT (curriculum-mutated) ones. Returns the bucket
    code per row (BUCKET_NAMES), or None when every probability is 0 (no random draw at all = the plain uniform sampler).

    One uniform r per row picks an exclusive bucket: [0, p_corner) corner, then slow, then spin, then single; the rest stays uniform.
    The RNG calls of corner / slow / spin are the pre-s3 code's, in its order; single draws only when single_prob > 0, so with
    single_prob 0 the RNG stream and the output are bit-identical to the pre-s3 sampler.

    single (s3, 2026-11-14): one axis a uniform among (vx, vy, wz), the other two exactly 0, sign 50/50, magnitude LOG-UNIFORM
        |c_a| = exp(ln m + u (ln E - ln m)),  u ~ U[0, 1),  m = single_min[a],  E = the range edge on that sign's side
    e.g. P(0.03 <= |vy| <= 0.10) = ln(0.10/0.03) / ln(0.3/0.03) = 0.52 -- the small commands the uniform draw almost never makes.
    An edge below m (a range not yet widened that far) gives |c_a| = E."""
    if corner_prob + slow_prob + spin_prob + single_prob <= 0.0:
        return None
    n, dev = cmd.shape[0], cmd.device
    r = torch.rand(n, device=dev)
    corner = r < corner_prob
    slow = (~corner) & (r < corner_prob + slow_prob)
    spin = (~corner) & (~slow) & (r < corner_prob + slow_prob + spin_prob)
    bucket = torch.zeros(n, dtype=torch.long, device=dev)
    bucket[corner] = 1
    bucket[slow] = 2
    bucket[spin] = 3
    # lm5e CORNER: every axis snapped to an edge of its current range, random sign per axis
    if corner.any():
        k = int(corner.sum())
        for axis, rng in enumerate((lin_vel_x, lin_vel_y, ang_vel_z)):
            hi_side = torch.rand(k, device=dev) < 0.5
            cmd[corner, axis] = torch.where(
                hi_side, torch.full_like(hi_side, rng[1], dtype=torch.float),
                torch.full_like(hi_side, rng[0], dtype=torch.float))
    # lm5g SLOW band (operator 2026-09-02: sub-0.2 striding is the weak zone in every lm5f run; "beautiful gait at small speeds ...
    # transfer to desk line"): lin vel redrawn inside +-slow_band, wz kept as drawn.
    if slow.any():
        k = int(slow.sum())
        band = float(slow_band or 0.3)
        for axis, rng in ((0, lin_vel_x), (1, lin_vel_y)):
            lo = max(rng[0], -band)
            hi = min(rng[1], band)
            cmd[slow, axis] = torch.empty(k, device=dev).uniform_(lo, hi)
    # lm5g SPIN (operator: "still no rotation while standing" on 2 of 3 lm5f runs; "make it worth rotating"): pure rotation in
    # place -- vx = vy = 0, wz drawn AWAY from zero (sign x U(spin_min, range hi)), not a dead-zone draw.
    if spin.any():
        k = int(spin.sum())
        cmd[spin, 0] = 0.0
        cmd[spin, 1] = 0.0
        wlo, whi = ang_vel_z
        mag_hi = max(abs(wlo), abs(whi))
        smin = min(float(spin_min or 0.2), mag_hi)
        mag = torch.empty(k, device=dev).uniform_(smin, mag_hi)
        sign = torch.where(torch.rand(k, device=dev) < 0.5, -1.0, 1.0)
        cmd[spin, 2] = mag * sign
    # s3 SINGLE-AXIS (2026-11-14: every policy stands still at small single-axis commands = a learned deadband; the uniform draw
    # makes (0, 0.05, 0) in ~0.15 % of draws)
    if single_prob > 0.0:
        single = (~corner) & (~slow) & (~spin) & (r < corner_prob + slow_prob + spin_prob + single_prob)
        if single.any():
            k = int(single.sum())
            assert min(single_min) > 0.0, "[cmd buckets] single_min must be > 0 (log-uniform)"
            axis = torch.randint(0, 3, (k,), device=dev)
            pos = torch.rand(k, device=dev) < 0.5
            u = torch.rand(k, device=dev)
            edges = torch.tensor([[lin_vel_x[1], lin_vel_y[1], ang_vel_z[1]],            # + side: the range's upper edge
                                  [-lin_vel_x[0], -lin_vel_y[0], -ang_vel_z[0]]],        # - side: |the lower edge|
                                 dtype=torch.float, device=dev)
            edge = torch.where(pos, edges[0, axis], edges[1, axis]).clamp(min=0.0)
            lo = torch.tensor([float(x) for x in single_min], dtype=torch.float, device=dev)[axis]
            hi = torch.maximum(edge, lo)
            mag = torch.minimum(torch.exp(torch.log(lo) + u * (torch.log(hi) - torch.log(lo))), edge)
            val = torch.zeros(k, 3, device=dev)
            val[torch.arange(k, device=dev), axis] = torch.where(pos, mag, -mag)
            cmd[single] = val
            bucket[single] = BUCKET_SINGLE + axis
    return bucket


class CornerVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand + CORNER SAMPLING (lm5e, operator 2026-08-28: "train
    multiple commanded velocities at once … 20-30 % of the time").

    Commands were always sampled jointly (uniform on all three axes per resample);
    what stays rare is the CORNERS — large magnitudes on several axes at once —
    because the per-axis curriculum widens each range on its own tracking score.
    With probability `corner_prob` a resample snaps every axis to an edge of its
    CURRENT range (random sign per axis), so back+left, fast+turn, … appear at
    full magnitude. Edges follow the curriculum (cfg.ranges is mutated by the
    level terms). Standing envs are still zeroed by _update_command.

    + slow / spin (lm5g) and single (s3) buckets: see sample_command_buckets. A single-axis draw is marked non-heading so
    _update_command keeps its wz (a heading env's wz is overwritten by the heading P-control -- as it is for corner / spin
    draws, unchanged)."""

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        c = self.cfg
        p = dict(corner_prob=float(getattr(c, "corner_prob", 0.0) or 0.0), slow_prob=float(getattr(c, "slow_prob", 0.0) or 0.0),
                 slow_band=float(getattr(c, "slow_band", 0.3) or 0.3), spin_prob=float(getattr(c, "spin_prob", 0.0) or 0.0),
                 spin_min=float(getattr(c, "spin_min", 0.2) or 0.2), single_prob=float(getattr(c, "single_prob", 0.0) or 0.0),
                 single_min=tuple(getattr(c, "single_min", (0.03, 0.03, 0.05))))
        if p["corner_prob"] + p["slow_prob"] + p["spin_prob"] + p["single_prob"] <= 0.0:
            return
        ids = torch.as_tensor(env_ids, device=self.device)
        cmd = self.vel_command_b[ids]                       # a copy (advanced indexing); written back below
        bucket = sample_command_buckets(cmd, c.ranges.lin_vel_x, c.ranges.lin_vel_y, c.ranges.ang_vel_z, **p)
        self.vel_command_b[ids] = cmd
        if p["single_prob"] > 0.0:
            self.is_heading_env[ids[bucket >= BUCKET_SINGLE]] = False
        self._bucket_proof(bucket, p)

    def _bucket_proof(self, bucket: torch.Tensor, p: dict):
        """Proof of life (JOB_VALIDITY 13): the realized bucket shares, printed ONCE after >= 2 x num_envs resamples."""
        if getattr(self, "_bucket_printed", False):
            return
        if getattr(self, "_bucket_counts", None) is None:
            self._bucket_counts = torch.zeros(len(BUCKET_NAMES), dtype=torch.long, device=self.device)
        self._bucket_counts += torch.bincount(bucket, minlength=len(BUCKET_NAMES))
        cnt = self._bucket_counts.tolist() if int(self._bucket_counts.sum()) >= 2 * self.num_envs else None
        if cnt is None:
            return
        self._bucket_printed = True
        n = sum(cnt)
        pc = [100.0 * x / n for x in cnt]
        p_uni = 1.0 - min(1.0, p["corner_prob"] + p["slow_prob"] + p["spin_prob"] + p["single_prob"])
        print(f"[cmd buckets] realized over {n} resamples (cfg in brackets): uniform {pc[0]:.1f} % ({100 * p_uni:.1f}), "
              f"corner {pc[1]:.1f} % ({100 * p['corner_prob']:.1f}), slow {pc[2]:.1f} % ({100 * p['slow_prob']:.1f}), "
              f"spin {pc[3]:.1f} % ({100 * p['spin_prob']:.1f}), single {sum(pc[4:]):.1f} % ({100 * p['single_prob']:.1f}) = "
              f"vx {pc[4]:.1f} / vy {pc[5]:.1f} / wz {pc[6]:.1f}; single_min {p['single_min']}, ranges vx {tuple(self.cfg.ranges.lin_vel_x)} "
              f"vy {tuple(self.cfg.ranges.lin_vel_y)} wz {tuple(self.cfg.ranges.ang_vel_z)}; standing envs zeroed downstream at "
              f"rel_standing_envs {self.cfg.rel_standing_envs}", flush=True)


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
    single_prob: float = 0.0
    """s3: fraction of resamples set to ONE axis (vx, vy or wz, 1/3 each), the other two 0, magnitude log-uniform between
    single_min[axis] and the current range edge on a random sign's side; non-heading. 0 = off (bit-identical to pre-s3)."""
    single_min: tuple = (0.03, 0.03, 0.05)
    """s3: the single bucket's smallest magnitude per axis (vx m/s, vy m/s, wz rad/s)."""
