from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


# --- OU IMU noise (HuB-style, p12g_imu_ou) -----------------------------------
# A per-env body-frame orientation-estimate error evolves as an OU process
# (mean-reverting random walk) once per policy step; projected gravity is
# derived from the PERTURBED quat and angular velocity is rotated into the
# same perturbed frame. This replaces independent per-term Unoise: real IMU
# error is temporally correlated and structurally coupled — one orientation
# error corrupts both quantities together. Params (std_deg = stationary
# per-axis std, tau_ms = correlation time) to be fitted from real FixStand
# lowstate logs; defaults are the HuB-implied fallback. Policy group only —
# the critic stays privileged/clean.

import math as _math

from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as _math_utils


def _imu_ou_err_quat(env: ManagerBasedRLEnv, std_deg: float, tau_ms: float) -> torch.Tensor:
    if not hasattr(env, "_imu_ou_rpy"):
        env._imu_ou_rpy = torch.zeros(env.num_envs, 3, device=env.device)
        env._imu_ou_step = -1
    if env._imu_ou_step != env.common_step_counter:
        env._imu_ou_step = env.common_step_counter
        dt = env.step_dt
        tau = tau_ms * 1e-3
        std_rad = std_deg * _math.pi / 180.0
        sw = std_rad * _math.sqrt(2.0 / tau)
        env._imu_ou_rpy += -(env._imu_ou_rpy / tau) * dt + sw * _math.sqrt(dt) * torch.randn_like(env._imu_ou_rpy)
    r = env._imu_ou_rpy
    return _math_utils.quat_from_euler_xyz(r[:, 0], r[:, 1], r[:, 2])


def projected_gravity_ou(
    env: ManagerBasedRLEnv,
    std_deg: float = 5.0,
    tau_ms: float = 40.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    err = _imu_ou_err_quat(env, std_deg, tau_ms)
    q_obs = _math_utils.quat_mul(asset.data.root_quat_w, err)
    return _math_utils.quat_apply_inverse(q_obs, asset.data.GRAVITY_VEC_W)


def base_ang_vel_ou(
    env: ManagerBasedRLEnv,
    std_deg: float = 5.0,
    tau_ms: float = 40.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    err = _imu_ou_err_quat(env, std_deg, tau_ms)
    return _math_utils.quat_apply_inverse(err, asset.data.root_ang_vel_b)


def anchor_point_b(
    env: ManagerBasedRLEnv,
    fwd_offset: float = 0.5,
    height_w: float = 1.0,
    noise_std: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DESK LINE v2: the anchor point in the BASE frame (3 floats).

    The anchor is a world-fixed point derived from spawn: spawn_xy + fwd_offset
    along the spawn heading, at ABSOLUTE world height height_w (the desk point,
    ~1.0 m). Deploy analog: the visual model publishes a desk point and the
    controller feeds it to the policy in the base frame — this obs is that
    contract, trained. noise_std adds gaussian jitter (robustness to a noisy
    detector). Needs capture_spawn_state buffers; zeros until they exist.
    """
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "spawn_root_xy") or not hasattr(env, "spawn_yaw"):
        return torch.zeros(env.num_envs, 3, device=env.device)
    fwd = torch.stack([torch.cos(env.spawn_yaw), torch.sin(env.spawn_yaw)], dim=-1)
    anchor_w = torch.cat(
        [env.spawn_root_xy + fwd_offset * fwd,
         torch.full((env.num_envs, 1), height_w, device=env.device)], dim=-1,
    )
    rel_b = _math_utils.quat_apply_inverse(
        asset.data.root_quat_w, anchor_w - asset.data.root_pos_w
    )
    if noise_std > 0.0:
        rel_b = rel_b + torch.randn_like(rel_b) * noise_std
    return rel_b


def anchor_point_b_held(
    env: ManagerBasedRLEnv,
    fwd_offset: float = 0.5,
    height_w: float = 1.0,
    noise_std: float = 0.0,
    hold_range_s: tuple = (0.5, 2.0),
    dropout_prob: float = 0.0,
    dropout_range_s: tuple = (2.0, 6.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """anchor_point_b as the REAL perception pipe delivers it (dp5h, hardware
    2026-09-07): SAMPLE-AND-HOLD in the BASE frame.

    On the robot the anchor is re-estimated only when the VisualModule
    processes a frame (~0.5-2 s apart) and MovementModule holds the last
    base-frame value in between — so while the body moves, the vector the
    policy sees is STALE (it drifts with the body until the next fix), then
    jumps. When detection drops out (tile occluded / out of view) the value
    freezes for seconds. dp5g_pushwin trained on a 50 Hz exact anchor and met
    this for the first time on the desk.

    Per env: refresh the held value from the live anchor_point_b (noise
    included) at reset and whenever its hold timer expires; each refresh draws
    the next hold from hold_range_s, or — with dropout_prob — a long freeze
    from dropout_range_s. State lives on the env (lazy) and is advanced at
    most once per env step, so a second obs group calling this term sees the
    same value. Use in the POLICY group only; the critic keeps the exact
    anchor_point_b (privileged, asymmetric)."""
    live = anchor_point_b(env, fwd_offset, height_w, noise_std, asset_cfg)
    n, dev = env.num_envs, env.device
    if not hasattr(env, "_anchor_held"):
        env._anchor_held = live.clone()
        env._anchor_hold_until = torch.zeros(n, dtype=torch.long, device=dev)
        env._anchor_hold_step = -1
        print(f"[anchor_point_b_held] policy anchor = sample-and-hold {hold_range_s[0]:.2f}-"
              f"{hold_range_s[1]:.2f} s, dropout p={dropout_prob:g} -> freeze "
              f"{dropout_range_s[0]:.1f}-{dropout_range_s[1]:.1f} s (step_dt {env.step_dt:.3f})", flush=True)
    step = int(env.common_step_counter)
    if env._anchor_hold_step != step:
        env._anchor_hold_step = step
        refresh = (env.episode_length_buf == 0) | (step >= env._anchor_hold_until)
        # dp8 (2026-09-09): expose this step's refresh mask so anchor_yaw_b_held
        # refreshes the SAME envs on the SAME tick — one VisualModule fix carries
        # both the table centroid and the Kabsch yaw.
        env._anchor_refresh_mask = refresh
        if refresh.any():
            idx = refresh.nonzero(as_tuple=False).squeeze(-1)
            env._anchor_held[idx] = live[idx]
            k = len(idx)
            hold_s = hold_range_s[0] + torch.rand(k, device=dev) * (hold_range_s[1] - hold_range_s[0])
            if dropout_prob > 0.0:
                drop = torch.rand(k, device=dev) < dropout_prob
                long_s = dropout_range_s[0] + torch.rand(k, device=dev) * (dropout_range_s[1] - dropout_range_s[0])
                hold_s = torch.where(drop, long_s, hold_s)
            env._anchor_hold_until[idx] = step + (hold_s / env.step_dt).round().long().clamp(min=1)
    return env._anchor_held


def _root_yaw(asset) -> torch.Tensor:
    q = asset.data.root_quat_w
    return torch.atan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]),
    )


def anchor_yaw_b(
    env: ManagerBasedRLEnv,
    noise_std: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """dp8 (operator 2026-09-09, 'I want the Kabsch yaw'): the TABLE's yaw
    relative to the base heading, as (cos, sin) of the error — 2 floats.

    The anchor vector tells the policy WHERE the table is; nothing in the 91-obs
    contract tells it how the table is TURNED (projected gravity has no yaw,
    the bearing to a point is not an orientation). On hardware the policy
    therefore drifts in yaw and stands skewed to the desk (09-07 run), while in
    sim heading_l2_from_spawn / anchor_hold(heading_to=spawn) shape a yaw it
    can only infer by integrating the gyro. This term is the missing channel.

    Sim source: spawn_yaw (the desk is placed along it; move_anchor with a
    yaw_range turns it). Deploy analog: DecisionModule's table_vector carries
    the Kabsch rotation of the 4 table markers vs their nominal poses; its yaw
    component, rotated to base frame, is this error. (cos, sin) rather than the
    raw angle: continuous through +-pi, bounded, no wrap-around edge for the
    MLP. Zeros-error = (1, 0) until the spawn buffers exist. noise_std is in
    radians on the error before the trig."""
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "spawn_yaw"):
        out = torch.zeros(env.num_envs, 2, device=env.device)
        out[:, 0] = 1.0
        return out
    err = env.spawn_yaw - _root_yaw(asset)
    if noise_std > 0.0:
        err = err + torch.randn_like(err) * noise_std
    return torch.stack([torch.cos(err), torch.sin(err)], dim=-1)


def anchor_yaw_b_held(
    env: ManagerBasedRLEnv,
    noise_std: float = 0.0,
    hold_range_s: tuple = (0.5, 2.0),
    dropout_prob: float = 0.0,
    dropout_range_s: tuple = (2.0, 6.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """anchor_yaw_b as the perception pipe delivers it: SAMPLE-AND-HOLD.

    Synchronised with anchor_point_b_held when that term is in the same obs
    group (evaluated earlier in cfg order): it refreshes exactly the envs the
    anchor refreshed this tick (env._anchor_refresh_mask), so the policy sees
    one coherent fix (centroid + yaw) per VisualModule frame, as on the robot
    (table_vector is one message). Falls back to its own schedule with the
    same parameters when the anchor is exact (no mask published this tick).
    Use in the POLICY group only; the critic keeps anchor_yaw_b."""
    live = anchor_yaw_b(env, noise_std, asset_cfg)
    n, dev = env.num_envs, env.device
    if not hasattr(env, "_anchor_yaw_held"):
        env._anchor_yaw_held = live.clone()
        env._anchor_yaw_hold_until = torch.zeros(n, dtype=torch.long, device=dev)
        env._anchor_yaw_hold_step = -1
        print(f"[anchor_yaw_b_held] policy table-yaw = sample-and-hold {hold_range_s[0]:.2f}-"
              f"{hold_range_s[1]:.2f} s, dropout p={dropout_prob:g}; synced to "
              f"anchor_point_b_held when present", flush=True)
    step = int(env.common_step_counter)
    if env._anchor_yaw_hold_step != step:
        env._anchor_yaw_hold_step = step
        synced = hasattr(env, "_anchor_refresh_mask") and getattr(env, "_anchor_hold_step", -2) == step
        if synced:
            refresh = env._anchor_refresh_mask
        else:
            refresh = (env.episode_length_buf == 0) | (step >= env._anchor_yaw_hold_until)
        if refresh.any():
            idx = refresh.nonzero(as_tuple=False).squeeze(-1)
            env._anchor_yaw_held[idx] = live[idx]
            if not synced:
                k = len(idx)
                hold_s = hold_range_s[0] + torch.rand(k, device=dev) * (hold_range_s[1] - hold_range_s[0])
                if dropout_prob > 0.0:
                    drop = torch.rand(k, device=dev) < dropout_prob
                    long_s = dropout_range_s[0] + torch.rand(k, device=dev) * (dropout_range_s[1] - dropout_range_s[0])
                    hold_s = torch.where(drop, long_s, hold_s)
                env._anchor_yaw_hold_until[idx] = step + (hold_s / env.step_dt).round().long().clamp(min=1)
    return env._anchor_yaw_held


def reach_point_b(
    env: ManagerBasedRLEnv,
    noise_std: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DP2B TILT2: the world-frame reach point (env.reach_point_w, resampled per
    episode by resample_reach_point) expressed in the BASE frame — the policy's
    perception of the target it should serve with a hand. Deploy analog: the
    vision stack publishes the point; the controller feeds it in base frame.
    Zeros until the buffer exists.
    """
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "reach_point_w"):
        return torch.zeros(env.num_envs, 3, device=env.device)
    rel_b = _math_utils.quat_apply_inverse(
        asset.data.root_quat_w, env.reach_point_w - asset.data.root_pos_w
    )
    if noise_std > 0.0:
        rel_b = rel_b + torch.randn_like(rel_b) * noise_std
    return rel_b


def arm_wish_b(
    env: ManagerBasedRLEnv,
    command_name: str = "arm_pose_command",
    noise_std: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """dp4c LEAN PROGRAM: the WISH — the world point each arm is trying to
    reach, in the BASE frame, both arms (6 floats: left xyz + right xyz).
    This is the step BEFORE resolution: the policy receives the intent even
    when the resolver cannot reach it, so leaning to close the world-frame gap
    is learnable AND expressible on deploy (the vision/ActionModule side
    publishes the same pre-resolution point). Zeros until the command exists.
    Deploy tail: additive, the anchor pattern."""
    asset = env.scene[asset_cfg.name]
    try:
        cmd = env.command_manager.get_term(command_name)
        wish_l, wish_r = cmd.wish_w["left"], cmd.wish_w["right"]
    except Exception:
        return torch.zeros(env.num_envs, 6, device=env.device)
    out = []
    for w in (wish_l, wish_r):
        rel = _math_utils.quat_apply_inverse(asset.data.root_quat_w, w - asset.data.root_pos_w)
        out.append(rel)
    rel_b = torch.cat(out, dim=-1)
    if noise_std > 0.0:
        rel_b = rel_b + torch.randn_like(rel_b) * noise_std
    return rel_b
