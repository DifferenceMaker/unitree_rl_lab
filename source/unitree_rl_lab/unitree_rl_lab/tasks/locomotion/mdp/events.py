from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def apply_sustained_external_force(
    env: "ManagerBasedRLEnv",
    env_ids: "Sequence[int]",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    force_magnitude_range: tuple = (10.0, 50.0),
    duration_range_s: tuple = (1.5, 4.0),
    max_force_n: float = 80.0,
):
    """Apply a sustained external force on a body for a random duration.
 
    Called via mode="interval" event term. Force is applied via
    set_external_force_and_torque, which persists until cleared. The
    companion event `clear_expired_sustained_pushes` clears forces when
    their duration elapses.
 
    Force direction is random horizontal (xy plane). Force magnitude
    and duration are sampled uniformly within the given ranges. Use
    the sustained_push_curriculum to ramp these ranges over training.
 
    Defensive guards (added 2026-05-24):
    - NaN/Inf check on sampled forces — skip if anything is non-finite
    - Hard cap at max_force_n (default 80N) regardless of curriculum
      magnitude_range — protects against runaway curricula or accidental
      misconfiguration
    """
    if force_magnitude_range[1] <= 0.0:
        return  # warmup level, no push
 
    asset: Articulation = env.scene[asset_cfg.name]
    body_name = asset_cfg.body_names
    if isinstance(body_name, list):
        body_name = body_name[0]
    body_id = asset.body_names.index(body_name)
 
    n = len(env_ids)
    angles = torch.rand(n, device=env.device) * 2 * math.pi
    magnitudes = torch.empty(n, device=env.device).uniform_(
        force_magnitude_range[0], force_magnitude_range[1]
    )
    durations = torch.empty(n, device=env.device).uniform_(
        duration_range_s[0], duration_range_s[1]
    )
 
    # Hard cap regardless of curriculum — runaway protection
    magnitudes = torch.clamp(magnitudes, max=max_force_n)
 
    forces = torch.zeros(n, 1, 3, device=env.device)
    forces[:, 0, 0] = magnitudes * torch.cos(angles)
    forces[:, 0, 1] = magnitudes * torch.sin(angles)
 
    # NaN/Inf guard — skip if anything looks broken
    if torch.isnan(forces).any() or torch.isinf(forces).any():
        print(f"[WARN] NaN/Inf in sustained push forces at step "
              f"{env.common_step_counter}, skipping this push event")
        return
 
    torques = torch.zeros_like(forces)
 
    env_ids_tensor = (
        torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        if not isinstance(env_ids, torch.Tensor) else env_ids
    )
    asset.set_external_force_and_torque(
        forces, torques,
        body_ids=[body_id],
        env_ids=env_ids_tensor,
    )
 
    # Lazy-init clear-time buffer (-1 = no active push)
    if not hasattr(env, "_sustained_push_clear_time"):
        env._sustained_push_clear_time = torch.full(
            (env.num_envs,), -1.0, device=env.device
        )
        env._sustained_push_body_id = body_id
 
    current_t = env.episode_length_buf[env_ids_tensor].float() * env.step_dt
    env._sustained_push_clear_time[env_ids_tensor] = current_t + durations


def clear_expired_sustained_pushes(
    env: "ManagerBasedRLEnv",
    env_ids: "Sequence[int]",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
):
    """Clear sustained pushes whose duration has elapsed.

    Run via mode="interval" with interval_range_s=(step_dt, step_dt) so
    it fires every policy step. Compares episode time to per-env clear
    time and zeros out the external force on expired envs.
    """
    if not hasattr(env, "_sustained_push_clear_time"):
        return

    current_t = env.episode_length_buf.float() * env.step_dt
    expired = (env._sustained_push_clear_time > 0) & (current_t >= env._sustained_push_clear_time)

    if not expired.any():
        return

    asset: Articulation = env.scene[asset_cfg.name]
    body_id = env._sustained_push_body_id

    expired_ids = expired.nonzero(as_tuple=True)[0]
    zero_forces = torch.zeros(len(expired_ids), 1, 3, device=env.device)
    asset.set_external_force_and_torque(
        zero_forces, zero_forces,
        body_ids=[body_id],
        env_ids=expired_ids,
    )
    env._sustained_push_clear_time[expired_ids] = -1.0

def capture_spawn_state(
    env: "ManagerBasedRLEnv",
    env_ids: "torch.Tensor",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
):
    """Capture per-env spawn root state + foot positions after reset.

    Stores three buffers on the env (lazy-initialized):
      env.spawn_root_xy   : (num_envs, 2)    spawn root xy in world frame
      env.spawn_yaw       : (num_envs,)      spawn yaw (z-axis rotation)
      env.spawn_foot_pos  : (num_envs, 2, 2) spawn foot xy for [L, R] feet

    Reward functions reference these buffers to penalize drift from spawn.
    Spawn state is captured AFTER reset (mode='reset'), so it includes any
    reset_base randomization (initial pose noise, etc.).

    This is a privileged signal — used in REWARDS only, never as policy obs.
    The real robot does not need to know its spawn position; the policy
    learned via these rewards just produces actions that minimize drift.
    """
    asset = env.scene[asset_cfg.name]

    # Lazy-init buffers attached to env
    if not hasattr(env, "spawn_root_xy"):
        env.spawn_root_xy = torch.zeros((env.num_envs, 2), device=env.device)
        env.spawn_yaw = torch.zeros(env.num_envs, device=env.device)
        env.spawn_foot_pos = torch.zeros((env.num_envs, 2, 2), device=env.device)

    # Root xy in world frame
    env.spawn_root_xy[env_ids] = asset.data.root_pos_w[env_ids, :2]

    # Yaw from quaternion (wxyz convention in Isaac Lab)
    quat = asset.data.root_quat_w[env_ids]
    yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    env.spawn_yaw[env_ids] = yaw

    # Foot xy — asset_cfg.body_ids resolved from regex at config time
    if asset_cfg.body_ids is not None and len(asset_cfg.body_ids) == 2:
        foot_pos = asset.data.body_pos_w[env_ids][:, asset_cfg.body_ids, :2]
        env.spawn_foot_pos[env_ids] = foot_pos