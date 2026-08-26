"""Safety terminations — the dp5 autopsy input-path guard #2.

A critic can only stay finite if the states it is asked to value stay sane.
``root_state_out_of_bounds`` terminates any env whose root state is non-finite
or physically absurd (a rigid body launched by a solver spike, a kinematic
interpenetration impulse, etc.) BEFORE those observations reach the rollout.
Zero future value is genuinely true there: no recovery exists from a 15 m/s
pelvis. NaN-safe by construction: comparisons with NaN are False, so the
explicit ``isfinite`` check does the catching that ``clamp`` never can.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_state_out_of_bounds(
    env: "ManagerBasedRLEnv",
    max_lin_vel: float = 15.0,
    max_ang_vel: float = 25.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True where the root state is non-finite or beyond physical bounds."""
    asset = env.scene[asset_cfg.name]
    root = asset.data.root_state_w  # (N, 13): pos 3, quat 4, lin vel 3, ang vel 3
    nonfinite = ~torch.isfinite(root).all(dim=-1)
    lin_over = torch.norm(asset.data.root_lin_vel_w, dim=-1) > max_lin_vel
    ang_over = torch.norm(asset.data.root_ang_vel_w, dim=-1) > max_ang_vel
    return nonfinite | lin_over | ang_over


def desk_penetration(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["pelvis", "torso_link"]),
    desk_name: str = "desk",
    xy_margin: float = 0.0,
    z_margin: float = 0.02,
) -> torch.Tensor:
    """dp6b (operator 2026-08-26): terminate the env when the robot is IN the desk.

    dp6_leancmd previews showed "almost half the robots get pushed inside the
    table, like a glitch push inside"; the -100 undesired_contacts term barely
    fired (-1.15/ep) because a body that has passed THROUGH the slab collider
    reports no contact — every step spent there is garbage data paid with legit
    rewards (anchor_hold, desk_reach, lean_track). Geometric check, no physics
    dependency: a listed body centre (pelvis, torso_link) whose xy lies inside
    the slab footprint (desk frame, so place_desk's per-env pose is honoured)
    with its z BELOW the desk top (+z_margin) is inside the table. Leaning OVER
    the desk (centre above the top) and standing NEXT to it (xy outside the
    footprint) never trigger. Weight-free termination: the unavoidable shove
    against the desk is not punished, only the impossible state is cut.
    """
    if desk_name not in env.scene.keys():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    asset = env.scene[asset_cfg.name]
    desk = env.scene[desk_name]
    size = getattr(getattr(desk.cfg, "spawn", None), "size", (0.5, 2.0, 0.05))
    hx, hy, hz = size[0] * 0.5 + xy_margin, size[1] * 0.5 + xy_margin, size[2] * 0.5
    dpos = desk.data.root_pos_w                     # (N, 3) slab centre
    dquat = desk.data.root_quat_w                   # (N, 4) wxyz
    # desk yaw (the slab is placed flat; only yaw matters)
    yaw = torch.atan2(2.0 * (dquat[:, 0] * dquat[:, 3] + dquat[:, 1] * dquat[:, 2]),
                      1.0 - 2.0 * (dquat[:, 2] ** 2 + dquat[:, 3] ** 2))
    c, s = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
    body = asset.data.body_pos_w[:, asset_cfg.body_ids, :]      # (N, B, 3)
    rel = body[:, :, :2] - dpos[:, None, :2]
    lx = rel[..., 0] * c + rel[..., 1] * s                        # into the desk frame
    ly = -rel[..., 0] * s + rel[..., 1] * c
    top = (dpos[:, 2] + hz).unsqueeze(1)
    inside = (lx.abs() < hx) & (ly.abs() < hy) & (body[:, :, 2] < top + z_margin)
    inside = inside & torch.isfinite(body).all(dim=-1)
    return inside.any(dim=1)
