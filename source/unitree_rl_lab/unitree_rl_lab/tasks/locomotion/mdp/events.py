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

def resample_reach_point(
    env: "ManagerBasedRLEnv",
    env_ids: "torch.Tensor",
    fwd_range: tuple = (0.30, 0.80),
    lat_range: tuple = (-0.30, 0.30),
    height_range: tuple = (0.90, 1.15),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """DP2B TILT2: sample a WORLD-frame reach point per episode (reset mode),
    in the spawn-yaw frame ahead of the spawn: forward 0.3-0.8 m (beyond the
    ~0.41 m upright arm reach by design — the hinge stimulus), lateral +-0.3,
    world height 0.9-1.15 (the 1 m desk band). Stored as env.reach_point_w for
    reach_point_b (obs) and hand_reach_bonus (reward). Requires
    capture_spawn_state (runs earlier in the same reset event list).
    """
    import torch
    if not hasattr(env, "reach_point_w"):
        env.reach_point_w = torch.zeros(env.num_envs, 3, device=env.device)
    if not hasattr(env, "spawn_root_xy"):
        return
    n = len(env_ids)
    dev = env.device
    fwd_d = fwd_range[0] + torch.rand(n, device=dev) * (fwd_range[1] - fwd_range[0])
    lat_d = lat_range[0] + torch.rand(n, device=dev) * (lat_range[1] - lat_range[0])
    h = height_range[0] + torch.rand(n, device=dev) * (height_range[1] - height_range[0])
    yaw = env.spawn_yaw[env_ids]
    c, s = torch.cos(yaw), torch.sin(yaw)
    env.reach_point_w[env_ids, 0] = env.spawn_root_xy[env_ids, 0] + fwd_d * c - lat_d * s
    env.reach_point_w[env_ids, 1] = env.spawn_root_xy[env_ids, 1] + fwd_d * s + lat_d * c
    env.reach_point_w[env_ids, 2] = h


def move_anchor(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    radius_range: tuple = (0.2, 0.4),
    yaw_range: tuple = (-0.4, 0.4),
    half_normal_sigma: float | None = None,
    linear_decay: bool = False,
):
    """dp4b (operator design 2026-08-05): mid-episode HOME RELOCATION.

    Shifts the whole spawn-anchored frame — spawn_root_xy, spawn_foot_pos and
    (optionally) spawn_yaw — by a random planar offset. Because the anchor
    obs/bonus AND the spawn-geography terms (heading/base_pos/foot_displacement
    -from-spawn) all reference these buffers, every authority relocates
    COHERENTLY: the anchor moves, and "home" moves with it — no tug-of-war.
    The policy must then close a 0.2-0.4 m error it can only fix by STEPPING;
    the anchor obs shows it exactly where to go. This trains the following
    behavior the desk line currently improvises out-of-distribution (training
    anchors never moved; sim2sim "wobbles toward the ball", 2026-08-05).

    Use as an interval event (e.g. every 4-8 s). No-op before the first spawn
    capture. Bounded by construction (radius_range clamps the jump)."""
    if not hasattr(env, "spawn_root_xy"):
        return
    n = len(env_ids)
    if linear_decay:
        # dp4c anchor-wander v2 (operator 2026-08-06): displacement density
        # decreasing LINEARLY from a peak at 0 to zero at radius_range[1] —
        # "[0;15] cm where 0 is the most likely outcome and 15 the least, in a
        # linear way". Inverse-CDF of the triangular(mode=0) distribution:
        # r = r_max * (1 - sqrt(U)).
        r = radius_range[1] * (1.0 - torch.sqrt(torch.rand(n, device=env.device)))
    elif half_normal_sigma is not None:
        # ANCHOR_WANDER spec, ported verbatim from the MuJoCo eval side
        # (anchor_pub.h): half-normal step |N(0, sigma)|, clamped at
        # radius_range[1] — models the real perception pipeline re-publishing
        # the desk point every 1-2 s with small re-estimates.
        r = torch.abs(torch.randn(n, device=env.device) * half_normal_sigma
                      ).clamp(max=radius_range[1])
    else:
        r = radius_range[0] + torch.rand(n, device=env.device) * (radius_range[1] - radius_range[0])
    th = torch.rand(n, device=env.device) * (2.0 * math.pi)
    d = torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=-1)
    env.spawn_root_xy[env_ids] += d
    env.spawn_foot_pos[env_ids] += d.unsqueeze(1)
    if yaw_range is not None:
        env.spawn_yaw[env_ids] += (yaw_range[0] + torch.rand(n, device=env.device)
                                   * (yaw_range[1] - yaw_range[0]))


def place_desk(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    fwd_offset: float = 0.5,
    top_z: float = 1.0,
    thickness: float = 0.05,
):
    """dp4c_deskcol: place the KINEMATIC desk slab in front of the spawn
    heading at reset (spawn yaw is per-env, so a static prim cannot align).
    Desk centre = spawn_xy + fwd_offset (the anchor point = desk centre by
    spec); top at top_z (the real 1.0 m). The balance policy finally FEELS
    the table — the 'IK resolves onto the desk, hand strikes it, robot
    stumbles viciously' failure mode becomes trainable."""
    if not hasattr(env, "spawn_root_xy"):
        return
    desk = env.scene["desk"]
    n = len(env_ids)
    fwd = torch.stack([torch.cos(env.spawn_yaw[env_ids]), torch.sin(env.spawn_yaw[env_ids])], dim=-1)
    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0:2] = env.spawn_root_xy[env_ids] + fwd_offset * fwd
    pose[:, 2] = top_z - thickness * 0.5
    half = env.spawn_yaw[env_ids] * 0.5
    pose[:, 3] = torch.cos(half)
    pose[:, 6] = torch.sin(half)
    desk.write_root_pose_to_sim(pose, env_ids=env_ids)
