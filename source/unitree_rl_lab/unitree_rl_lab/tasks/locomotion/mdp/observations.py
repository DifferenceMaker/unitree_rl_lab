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
