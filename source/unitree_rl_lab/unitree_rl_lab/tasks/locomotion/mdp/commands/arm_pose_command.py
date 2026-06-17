"""Arm pose command for option (ii) — Phase 4 + p9 envelope expansion.

The arms are NOT policy-controlled; this command term drives them externally
and the policy observes the commanded held pose (14-dim). Each step:

    target = held_pose + wobble_amplitude * sin(2*pi * freq * t + phase)

Each episode samples (per-env, per-DOF) an amplitude ~ U(0, max) and then a
held delta within it, so the batch always spans "stand still" -> "max
disturbance" (preserved from v4.1 — see _resample_command).

p9 EXPANSION (all OFF by default -> reproduces p8_gold exactly):
- shoulder_yaw added as a sampled DOF (yaw_amplitude, default 0).
- decoupled left/right sampling via a blend fraction decouple_lr in [0,1]:
  0 = bilateral pitch/elbow + mirrored roll (gold), 1 = fully independent arms.
- overhead pitch: pitch_overhead_amplitude extends the pitch range in the
  pitch_overhead_sign direction (default negative = arm raised).
- self-collision clamps (roll_clamp, yaw_clamp, pitch_overhead_clamp) + a
  blanket clamp of every target to the URDF soft joint limits.

At the defaults (yaw_amplitude=0, decouple_lr=0, pitch_overhead_amplitude=0)
the sampler is identical to p8_gold. The curriculum ramps these knobs up from
level 0 (=gold) so a gold warmstart sees no envelope shock at iter 0.

Sampled DOF -> wobble channels (wobble is NOT in the obs; small reactive
disturbance only): pitch (both arms, ch0), left_roll (ch1), right_roll (ch2),
elbow (both arms, ch3). Yaw has no wobble channel (held only).
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Per-arm regexes so left/right can be sampled independently (decoupling).
LEFT_SHOULDER_PITCH_REGEX = ["left_shoulder_pitch_joint"]
RIGHT_SHOULDER_PITCH_REGEX = ["right_shoulder_pitch_joint"]
LEFT_SHOULDER_ROLL_REGEX = ["left_shoulder_roll_joint"]
RIGHT_SHOULDER_ROLL_REGEX = ["right_shoulder_roll_joint"]
LEFT_SHOULDER_YAW_REGEX = ["left_shoulder_yaw_joint"]
RIGHT_SHOULDER_YAW_REGEX = ["right_shoulder_yaw_joint"]
LEFT_ELBOW_PITCH_REGEX = ["left_elbow_pitch_joint"]
RIGHT_ELBOW_PITCH_REGEX = ["right_elbow_pitch_joint"]


class UniformArmPoseCommand(CommandTerm):
    """Episode-stable held pose + sinusoidal wobble per joint, with optional
    decoupled L/R sampling, shoulder-yaw, and overhead pitch (p9 expansion).

    See module docstring. With the expansion knobs at their defaults this
    reproduces p8_gold (bilateral pitch/elbow, mirrored roll, no yaw).
    """

    cfg: "UniformArmPoseCommandCfg"

    def __init__(self, cfg: "UniformArmPoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self.all_arm_joint_ids, self.all_arm_joint_names = self.robot.find_joints(cfg.all_arm_joint_names)
        self.num_arm_joints = len(self.all_arm_joint_ids)

        # Per-arm joint ids for each sampled DOF.
        self.lp_jid, _ = self.robot.find_joints(LEFT_SHOULDER_PITCH_REGEX)
        self.rp_jid, _ = self.robot.find_joints(RIGHT_SHOULDER_PITCH_REGEX)
        self.lr_jid, _ = self.robot.find_joints(LEFT_SHOULDER_ROLL_REGEX)
        self.rr_jid, _ = self.robot.find_joints(RIGHT_SHOULDER_ROLL_REGEX)
        self.ly_jid, _ = self.robot.find_joints(LEFT_SHOULDER_YAW_REGEX)
        self.ry_jid, _ = self.robot.find_joints(RIGHT_SHOULDER_YAW_REGEX)
        self.le_jid, _ = self.robot.find_joints(LEFT_ELBOW_PITCH_REGEX)
        self.re_jid, _ = self.robot.find_joints(RIGHT_ELBOW_PITCH_REGEX)

        # Index of each joint within the 14-dim arm observation vector.
        idx = lambda jid: self.all_arm_joint_ids.index(jid[0])
        self.lp_obs, self.rp_obs = idx(self.lp_jid), idx(self.rp_jid)
        self.lr_obs, self.rr_obs = idx(self.lr_jid), idx(self.rr_jid)
        self.ly_obs, self.ry_obs = idx(self.ly_jid), idx(self.ry_jid)
        self.le_obs, self.re_obs = idx(self.le_jid), idx(self.re_jid)

        # Per-env per-channel phase offsets for wobble (4 channels:
        # pitch, left_roll, right_roll, elbow). Yaw has no wobble.
        self.wobble_phase = torch.zeros(env.num_envs, 4, device=env.device)
        self.wobble_amp_episode = torch.zeros(env.num_envs, device=env.device)

        # Per-env time tracking for wobble (reset on episode reset).
        self.t = torch.zeros(env.num_envs, device=env.device)

        # 14-dim observation: held pose only (wobble excluded).
        self.command_b = torch.zeros(env.num_envs, self.num_arm_joints, device=env.device)

        # Default positions per joint group (num_envs, 1).
        dpos = self.robot.data.default_joint_pos
        self.def_lp = dpos[:, self.lp_jid].clone()
        self.def_rp = dpos[:, self.rp_jid].clone()
        self.def_lr = dpos[:, self.lr_jid].clone()
        self.def_rr = dpos[:, self.rr_jid].clone()
        self.def_ly = dpos[:, self.ly_jid].clone()
        self.def_ry = dpos[:, self.ry_jid].clone()
        self.def_le = dpos[:, self.le_jid].clone()
        self.def_re = dpos[:, self.re_jid].clone()
        self.default_arm_pos = dpos[:, self.all_arm_joint_ids].clone()

        # Soft joint position limits per group (num_envs, 1), with margin.
        self.soft_lp = self._soft_limits(self.lp_jid)
        self.soft_rp = self._soft_limits(self.rp_jid)
        self.soft_lr = self._soft_limits(self.lr_jid)
        self.soft_rr = self._soft_limits(self.rr_jid)
        self.soft_ly = self._soft_limits(self.ly_jid)
        self.soft_ry = self._soft_limits(self.ry_jid)
        self.soft_le = self._soft_limits(self.le_jid)
        self.soft_re = self._soft_limits(self.re_jid)

        # Step dt for time integration.
        self.dt = env.step_dt

    def _soft_limits(self, jids):
        """Return (lo, hi) soft joint-pos limits for a joint group, shrunk by
        cfg.soft_limit_margin. Each is shape (num_envs, len(jids))."""
        lims = self.robot.data.soft_joint_pos_limits[:, jids, :]
        lo = lims[..., 0] + self.cfg.soft_limit_margin
        hi = lims[..., 1] - self.cfg.soft_limit_margin
        return lo, hi

    def __str__(self) -> str:
        return (
            f"UniformArmPoseCommand(held pose + wobble + p9 expansion, "
            f"pitch_amp={self.cfg.pitch_amplitude:.2f}, roll_amp={self.cfg.roll_amplitude:.2f}, "
            f"elbow_amp={self.cfg.elbow_amplitude:.2f}, yaw_amp={self.cfg.yaw_amplitude:.2f}, "
            f"decouple_lr={self.cfg.decouple_lr:.2f}, "
            f"pitch_overhead_amp={self.cfg.pitch_overhead_amplitude:.2f}, "
            f"wobble_amp={self.cfg.wobble_amplitude:.2f}@{self.cfg.wobble_frequency:.1f}Hz)"
        )

    @property
    def command(self) -> torch.Tensor:
        """14-dim observation: held pose only (no wobble in obs)."""
        return self.default_arm_pos + self.command_b

    def _sample_pair(self, n, amp_ep, bilateral_sign, f, overhead_amp=None, overhead_sign=-1.0):
        """Sample a (left, right) held-delta pair for one DOF.

        amp_ep: per-env episode amplitude (n,). bilateral_sign: +1 if both arms
        share the delta (pitch/elbow/yaw), -1 if mirrored (roll). f: decouple
        fraction in [0,1] — 0 reproduces gold (left=shared, right=sign*shared),
        1 = independent arms. overhead_amp: per-env extra range added on the
        overhead_sign side (pitch only).
        """
        lo = -amp_ep
        hi = amp_ep
        if overhead_amp is not None:
            if overhead_sign < 0:
                lo = lo - overhead_amp
            else:
                hi = hi + overhead_amp

        def draw():
            return lo + torch.rand(n, device=self.device) * (hi - lo)

        shared = draw()
        left = shared + f * (draw() - shared)
        right_base = bilateral_sign * shared
        right = right_base + f * (draw() - right_base)
        return left, right

    def _resample_command(self, env_ids):
        """Resample episode amplitudes, held deltas, and wobble phase at reset."""
        n = len(env_ids)
        if n == 0:
            return

        r = lambda: torch.rand(n, device=self.device)
        # Per-episode per-env amplitudes ~ U(0, max) — preserves the
        # "stand still -> max disturbance" spectrum each batch.
        pitch_amp = r() * self.cfg.pitch_amplitude
        roll_amp = r() * self.cfg.roll_amplitude
        elbow_amp = r() * self.cfg.elbow_amplitude
        yaw_amp = r() * self.cfg.yaw_amplitude
        overhead_amp = r() * self.cfg.pitch_overhead_amplitude
        self.wobble_amp_episode[env_ids] = r() * self.cfg.wobble_amplitude

        f = float(self.cfg.decouple_lr)
        # pitch/elbow/yaw are bilateral (sign +1); roll is mirrored (sign -1).
        lp, rp = self._sample_pair(n, pitch_amp, +1.0, f, overhead_amp, self.cfg.pitch_overhead_sign)
        lr, rr = self._sample_pair(n, roll_amp, -1.0, f)
        le, re = self._sample_pair(n, elbow_amp, +1.0, f)
        ly, ry = self._sample_pair(n, yaw_amp, +1.0, f)

        # Wobble phase offsets uniform in [0, 2pi]; reset time.
        self.wobble_phase[env_ids] = torch.rand(n, 4, device=self.device) * (2.0 * math.pi)
        self.t[env_ids] = 0.0

        # Write held deltas into the observation vector (held pose only).
        self.command_b[env_ids] = 0.0
        self.command_b[env_ids, self.lp_obs] = lp
        self.command_b[env_ids, self.rp_obs] = rp
        self.command_b[env_ids, self.lr_obs] = lr
        self.command_b[env_ids, self.rr_obs] = rr
        self.command_b[env_ids, self.le_obs] = le
        self.command_b[env_ids, self.re_obs] = re
        self.command_b[env_ids, self.ly_obs] = ly
        self.command_b[env_ids, self.ry_obs] = ry

    def _apply_joint(self, jids, obs_idx, default_pos, soft_lim, wobble_col=None, clamp_val=None):
        """Apply held(+wobble) offset to one joint group as a position target,
        with optional self-collision clamp and the soft-limit clamp."""
        offset = self.command_b[:, obs_idx]                       # (num_envs,)
        if wobble_col is not None:
            offset = offset + self._wobble[:, wobble_col]
        if clamp_val is not None:
            offset = torch.clamp(offset, -clamp_val, clamp_val)
        target = default_pos + offset.unsqueeze(1)                # (num_envs, 1)
        if self.cfg.clamp_to_soft_limits:
            target = torch.clamp(target, soft_lim[0], soft_lim[1])
        self.robot.set_joint_position_target(target, joint_ids=jids)

    def _update_command(self):
        """Apply held pose + wobble each step, with collision + soft-limit clamps."""
        if not self.cfg.apply_directly:
            return

        # Advance time, compute the 4 wobble channels (per-env).
        self.t += self.dt
        omega_t = 2.0 * math.pi * self.cfg.wobble_frequency * self.t
        self._wobble = self.wobble_amp_episode.unsqueeze(1) * torch.sin(
            omega_t.unsqueeze(1) + self.wobble_phase
        )  # (num_envs, 4): pitch, left_roll, right_roll, elbow

        pc = self.cfg.pitch_overhead_clamp
        rc = self.cfg.roll_clamp
        yc = self.cfg.yaw_clamp

        # Pitch (both arms, wobble ch0, overhead clamp).
        self._apply_joint(self.lp_jid, self.lp_obs, self.def_lp, self.soft_lp, wobble_col=0, clamp_val=pc)
        self._apply_joint(self.rp_jid, self.rp_obs, self.def_rp, self.soft_rp, wobble_col=0, clamp_val=pc)
        # Roll (left ch1 / right ch2, +/- roll_clamp — preserves p7_1b guard).
        self._apply_joint(self.lr_jid, self.lr_obs, self.def_lr, self.soft_lr, wobble_col=1, clamp_val=rc)
        self._apply_joint(self.rr_jid, self.rr_obs, self.def_rr, self.soft_rr, wobble_col=2, clamp_val=rc)
        # Elbow (both arms, wobble ch3, no collision clamp — soft limit only).
        self._apply_joint(self.le_jid, self.le_obs, self.def_le, self.soft_le, wobble_col=3, clamp_val=None)
        self._apply_joint(self.re_jid, self.re_obs, self.def_re, self.soft_re, wobble_col=3, clamp_val=None)
        # Yaw (held only, no wobble, +/- yaw_clamp).
        self._apply_joint(self.ly_jid, self.ly_obs, self.def_ly, self.soft_ly, wobble_col=None, clamp_val=yc)
        self._apply_joint(self.ry_jid, self.ry_obs, self.def_ry, self.soft_ry, wobble_col=None, clamp_val=yc)

    def _update_metrics(self):
        pass


@configclass
class UniformArmPoseCommandCfg(CommandTermCfg):
    """Cfg for UniformArmPoseCommand (Phase 4 + p9 expansion).

    Expansion knobs default to gold-equivalent (yaw off, no decouple, no
    overhead) so the base task reproduces p8_gold; the curriculum ramps them.
    """

    class_type: type = UniformArmPoseCommand

    asset_name: str = MISSING
    all_arm_joint_names: list[str] = MISSING

    # --- held-pose amplitudes (max of the per-episode U(0,max) draw) ---
    pitch_amplitude: float = 0.0
    """Max |shoulder_pitch delta| (rad). Gold: 1.5 (set by curriculum)."""

    roll_amplitude: float = 0.0
    """Max |shoulder_roll delta| (rad), mirrored L/R at decouple_lr=0. Gold: 1.0."""

    elbow_amplitude: float = 0.0
    """Max |elbow_pitch delta| (rad). Gold: 1.5."""

    yaw_amplitude: float = 0.0
    """Max |shoulder_yaw delta| (rad). p9: enables pointing. Gold: 0.0 (off)."""

    # --- p9 expansion controls (all gold-off by default) ---
    decouple_lr: float = 0.0
    """L/R sampling blend in [0,1]: 0 = bilateral pitch/elbow + mirrored roll
    (gold), 1 = fully independent arms. Ramped by the curriculum."""

    pitch_overhead_amplitude: float = 0.0
    """Extra pitch range (rad) added in the pitch_overhead_sign direction, to
    reach above-shoulder poses. Gold: 0.0. Per-episode U(0,max) like the rest."""

    pitch_overhead_sign: float = -1.0
    """Sign of the 'overhead' pitch direction. -1 => negative pitch raises the
    arm (assumed for H1_2; flip to +1 if a sim check shows otherwise)."""

    # --- self-collision / safety clamps (delta from default, rad) ---
    roll_clamp: float = 0.8
    """Clamp on shoulder_roll offset (held+wobble). Was hardcoded ±0.8 (p7_1b)."""

    yaw_clamp: float = 0.8
    """Clamp on shoulder_yaw offset. Conservative start — FLAG FOR REVIEW."""

    pitch_overhead_clamp: float = 2.0
    """Clamp on shoulder_pitch offset magnitude (held+wobble). 2.0 > gold's 1.5
    so it does not bite gold; bounds overhead reach. FLAG FOR REVIEW."""

    clamp_to_soft_limits: bool = True
    """Clamp every commanded target to the URDF soft joint-pos limits."""

    soft_limit_margin: float = 0.02
    """Margin (rad) kept inside each soft joint limit when clamping targets."""

    # --- wobble (validated; do not increase amplitude past 0.25) ---
    wobble_amplitude: float = 0.0
    """Sinusoidal wobble amplitude (rad) on top of the held pose. Set by curriculum."""

    wobble_frequency: float = 2.0
    """Wobble frequency in Hz."""

    apply_directly: bool = True
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    debug_vis: bool = False
