# Ported from Amazon FAR's holosoma (Apache-2.0):
#   managers/reward/terms/locomotion.py + managers/command/terms/locomotion.py
# (LocomotionGait phase machinery folded into feet_phase — IsaacLab has no
# command-term slot for a pure phase clock, so the per-env phase state lives
# on the env object instead).
#
# These are the FastSAC Run B rewards: the paper's minimalist locomotion set,
# math verbatim. Terms holosoma covers with stock IsaacLab funcs (tracking,
# ang_vel_xy_l2, flat_orientation_l2, action_rate_l2, is_alive) are NOT
# duplicated here — the env cfg maps those directly. Sigma convention note:
# holosoma tracking exp(-err/0.25) == IsaacLab exp(-err/std^2) at std=0.5.
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _bezier_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Expected foot height from gait phase using a cubic Bezier profile (verbatim)."""

    def cubic_bezier(y_start, y_end, x):
        return y_start + (y_end - y_start) * (x**3 + 3 * (x**2 * (1 - x)))

    x = (phi + math.pi) / (2 * math.pi)
    stance = cubic_bezier(torch.zeros_like(x), torch.full_like(x, swing_height), 2 * x)
    swing = cubic_bezier(torch.full_like(x, swing_height), torch.zeros_like(x), 2 * x - 1)
    return torch.where(x <= 0.5, stance, swing)


class _GaitPhaseState:
    """Per-env phase clock replicating holosoma's LocomotionGait command term:
    random per-episode phase offset (feet antiphase), gait frequency sampled
    in [1/period - width, 1/period + width], phase forced to pi (both feet
    expected planted) while the command is ~zero."""

    def __init__(self, env, period: float, rand_width: float):
        n, dev = env.num_envs, env.device
        self.period = period
        self.rand_width = rand_width
        self.offset = torch.zeros(n, 2, device=dev)
        self.freq = torch.full((n, 1), 1.0 / period, device=dev)

    def resample(self, env, env_ids: torch.Tensor):
        n = len(env_ids)
        if n == 0:
            return
        off0 = torch.rand(n, device=env.device) * 2 * math.pi - math.pi
        self.offset[env_ids, 0] = off0
        self.offset[env_ids, 1] = torch.fmod(off0 + 2 * math.pi, 2 * math.pi) - math.pi
        base = 1.0 / self.period
        if self.rand_width > 0:
            self.freq[env_ids, 0] = base + (torch.rand(n, device=env.device) * 2 - 1) * self.rand_width
        else:
            self.freq[env_ids, 0] = base

    def phase(self, env, command_name: str) -> torch.Tensor:
        # resample rows that just reset (episode clock at 0)
        fresh = (env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
        if fresh.numel():
            self.resample(env, fresh)
        phase_dt = 2 * math.pi * env.step_dt * self.freq
        raw = env.episode_length_buf.unsqueeze(1) * phase_dt + self.offset
        phase = torch.fmod(raw + math.pi, 2 * math.pi) - math.pi
        cmd = env.command_manager.get_command(command_name)
        stand = (torch.linalg.norm(cmd[:, :2], dim=1) < 0.01) & (cmd[:, 2].abs() < 0.01)
        phase[stand] = math.pi
        return phase


def feet_phase(
    env: "ManagerBasedRLEnv",
    command_name: str,
    std: float = 0.008,
    swing_height: float = 0.09,
    period: float = 1.0,
    period_rand_width: float = 0.2,
    foot_rest_z: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
) -> torch.Tensor:
    """holosoma feet_phase: exp(-(errL+errR)/std) on Bezier swing-height tracking.

    foot_rest_z: z of the foot body origin when the sole is on flat ground —
    their feet_heights are terrain-relative, ours come from the ankle_roll
    body origin, so the rest offset is subtracted.
    """
    state = getattr(env, "_hs_gait_state", None)
    if state is None or state.offset.shape[0] != env.num_envs:
        state = _GaitPhaseState(env, period, period_rand_width)
        env._hs_gait_state = state
    phase = state.phase(env, command_name)

    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_rest_z  # (n, 2)
    expected = _bezier_foot_height(phase, swing_height)
    err = torch.sum(torch.square(foot_z - expected), dim=1)
    return torch.exp(-err / std)


def pose_deviation_weighted(
    env: "ManagerBasedRLEnv",
    weights: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """holosoma pose: weighted squared deviation from the default pose.

    weights: {joint-name regex: weight}; every actuated joint must be covered
    by exactly one pattern (checked once, cached on the env).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wvec = getattr(env, "_hs_pose_weights", None)
    if wvec is None:
        wvec = torch.zeros(asset.num_joints, device=env.device)
        covered = torch.zeros(asset.num_joints, dtype=torch.bool, device=env.device)
        for pattern, w in weights.items():
            ids, _ = asset.find_joints(pattern)
            for i in ids:
                if covered[i]:
                    raise ValueError(f"pose weight pattern '{pattern}' re-covers joint {i}")
            wvec[ids] = w
            covered[ids] = True
        if not covered.all():
            missing = [asset.joint_names[i] for i in (~covered).nonzero().squeeze(-1).tolist()]
            raise ValueError(f"pose weights leave joints uncovered: {missing}")
        env._hs_pose_weights = wvec
    err = torch.square(asset.data.joint_pos - asset.data.default_joint_pos)
    return torch.sum(err * wvec.unsqueeze(0), dim=1)


def close_feet_lateral(
    env: "ManagerBasedRLEnv",
    close_feet_threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
) -> torch.Tensor:
    """holosoma penalty_close_feet_xy: binary penalty when the lateral
    (base-yaw-perpendicular) distance between the feet is under threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]  # (n, 2, 2)
    # base yaw from the yaw-only quaternion applied to +x
    yaw_q = yaw_quat(asset.data.root_quat_w)
    x_axis = torch.zeros(env.num_envs, 3, device=env.device)
    x_axis[:, 0] = 1.0
    base_fwd = quat_apply(yaw_q, x_axis)
    base_yaw = torch.atan2(base_fwd[:, 1], base_fwd[:, 0])
    d = torch.abs(
        torch.cos(base_yaw) * (feet_xy[:, 0, 1] - feet_xy[:, 1, 1])
        - torch.sin(base_yaw) * (feet_xy[:, 0, 0] - feet_xy[:, 1, 0])
    )
    return (d < close_feet_threshold).float()


def feet_flat_orientation(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
) -> torch.Tensor:
    """holosoma penalty_feet_ori: sum over feet of ||gravity_xy in foot frame||."""
    asset: Articulation = env.scene[asset_cfg.name]
    quats = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (n, 2, 4)
    g = torch.zeros(env.num_envs, 3, device=env.device)
    g[:, 2] = -1.0
    out = torch.zeros(env.num_envs, device=env.device)
    for f in range(quats.shape[1]):
        g_foot = quat_apply_inverse(quats[:, f], g)
        out += torch.sum(torch.square(g_foot[:, :2]), dim=1) ** 0.5
    return out


# ── penalty curriculum (holosoma PenaltyCurriculum + AverageEpisodeLengthTracker) ──

def penalty_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
    term_names: list[str],
    initial_scale: float = 0.5,
    min_scale: float = 0.5,
    max_scale: float = 1.0,
    level_down_threshold: float = 150.0,
    level_up_threshold: float = 750.0,
    degree: float = 0.001,
    num_compute_average_epl: int = 1000,
) -> float:
    """holosoma's penalty ramp, verbatim semantics: penalty weights start at
    initial_scale x nominal; a running average of finished-episode length
    (EMA over ~num_compute_average_epl episodes) moves the scale
    multiplicatively by `degree` per reset call — down while episodes are
    short (< level_down), up once they are long (> level_up) — clamped to
    [min_scale, max_scale]. Returns the current scale (logged as
    Curriculum/penalty_curriculum)."""
    st = getattr(env, "_hs_penalty_state", None)
    if st is None:
        st = {"scale": float(initial_scale), "avg_len": 0.0,
              "orig": {n: float(env.reward_manager.get_term_cfg(n).weight) for n in term_names}}
        for n in term_names:
            cfg = env.reward_manager.get_term_cfg(n)
            cfg.weight = st["orig"][n] * st["scale"]
            env.reward_manager.set_term_cfg(n, cfg)
        env._hs_penalty_state = st
        return st["scale"]

    if len(env_ids) > 0:
        # tracker: EMA over the episodes that just ended (their lengths are the
        # env clocks at reset time); alpha = n_ended / num_compute_average_epl
        ended = env.episode_length_buf[env_ids].float().mean().item()
        alpha = min(1.0, len(env_ids) / max(1, num_compute_average_epl))
        st["avg_len"] = (1 - alpha) * st["avg_len"] + alpha * ended

    if st["avg_len"] < level_down_threshold:
        st["scale"] *= 1.0 - degree
    elif st["avg_len"] > level_up_threshold:
        st["scale"] *= 1.0 + degree
    st["scale"] = float(min(max(st["scale"], min_scale), max_scale))

    for n in term_names:
        cfg = env.reward_manager.get_term_cfg(n)
        cfg.weight = st["orig"][n] * st["scale"]
        env.reward_manager.set_term_cfg(n, cfg)
    return st["scale"]
