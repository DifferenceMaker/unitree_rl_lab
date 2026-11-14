"""LM7I terms (2026-11-01): the three focus items on lm7h_b_h4_still -- no counter-sway after a push, arms locked, calm recoveries.

Operator 2026-10-31 sim2sim: lm7h_a_h4_still "coutner sways when pushed like mad", lm7h_b_h4_still "I hate the anti-swaying a lot";
"Counter-swaying should be taken as most possible from our p14h_stancewide".

  track_lin_vel_xy_split   the walk's velocity-tracking kernel with a WIDE kernel at a zero command (the p14h form). The parent pays
                           20 exp(-|v_err|^2 / std^2) with std on the tracking ladder (.15 -> .10 -> .05; lm7h_b_h4_still ENDED at .05):
                           at a zero command a 0.3 m/s push forfeits all 20 per step at std .05 (19.6 at .15), so the cheapest answer
                           to a shove is the fastest arrest -- lean into it, throw a foot, overshoot = the counter-sway (p14_recovery's
                           cost read). p14h_stancewide pays 1.5 exp(-|v|^2 / 0.5^2) (forfeits 0.45 of 1.5). Gated on the COMMAND (in the
                           observation; Bible rule 32): |cmd| < cmd_thr -> std_stand, else the ladder's std (the curriculum keeps
                           mutating params["std"]; std_stand is a separate param it never touches).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cmd_relative_std(std: float, cmd_mag: torch.Tensor, cref: float, fmin: float) -> torch.Tensor:
    """s3 command-relative kernel width (2026-11-14): sigma_eff = std x clamp(|c| / cref, fmin, 1). At a small command the kernel
    narrows with the command, so standing still stops paying most of the income: sigma 0.0817, cref 0.16, |c| 0.05 -> sigma_eff
    0.0817 x 0.3125 = 0.0255 and exp(-(0.05 / 0.0255)^2) = 0.022 instead of exp(-(0.05 / 0.0817)^2) = 0.69. |c| >= cref -> std
    itself; fmin floors the narrowing (a zero command component -> std x fmin)."""
    return std * torch.clamp(cmd_mag / cref, min=fmin, max=1.0)


def track_lin_vel_xy_split(
    env: "ManagerBasedRLEnv",
    std: float,
    std_stand: float = 0.5,
    command_name: str = "base_velocity",
    cmd_thr: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cref: float = 0.0,
    fmin: float = 0.25,
) -> torch.Tensor:
    """isaaclab's track_lin_vel_xy_yaw_frame_exp (velocity error in the gravity-aligned yaw frame) with kernel std_stand while the
    full command norm (vx, vy, wz) is below cmd_thr -- the line's standing gate -- and the tracking ladder's std otherwise. At a
    standstill both branches pay the full weight (error 0); after a push the standing branch keeps paying: 20 exp(-0.3^2 / 0.5^2)
    = 13.95 at 0.3 m/s instead of ~0 at std .05.
    s3 (cref > 0): the moving branch's std becomes cmd_relative_std(std, |(vx, vy)_cmd|, cref, fmin); std is the CURRENT param
    (the tracking_std_levels ladder mutates it). cref 0 (the default, and every cfg written before s3) = the ops above, unchanged."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    cmd = env.command_manager.get_command(command_name)
    err = torch.sum(torch.square(cmd[:, :2] - vel_yaw[:, :2]), dim=1)
    standing = torch.linalg.norm(cmd, dim=1) < cmd_thr
    if cref > 0.0:
        s_move = cmd_relative_std(std, torch.linalg.norm(cmd[:, :2], dim=1), cref, fmin)
    else:
        s_move = torch.full_like(err, std)
    s = torch.where(standing, torch.full_like(err, std_stand), s_move)
    return torch.exp(-err / torch.square(s))


def track_ang_vel_z_exp_rel(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cref: float = 0.0,
    fmin: float = 0.25,
    cmd_thr: float = 0.03,
) -> torch.Tensor:
    """isaaclab's track_ang_vel_z_exp (exp(-(wz_cmd - wz_body)^2 / std^2)) + the s3 command-relative width: for MOVING envs (full
    command norm >= cmd_thr, the split term's gate) std -> cmd_relative_std(std, |wz_cmd|, cref, fmin); standing envs keep std
    (the parent's term has no standing branch -- its standing envs are left exactly as they were). cref 0 = isaaclab's ops,
    bit-identical; with cref > 0 every row the narrowing does not touch (standing, or |wz_cmd| >= cref) keeps isaaclab's value
    to the bit (the narrowed rows are swapped in with torch.where)."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])
    base = torch.exp(-ang_vel_error / std**2)
    if cref <= 0.0:
        return base
    wz = cmd[:, 2].abs()
    narrow = (torch.linalg.norm(cmd, dim=1) >= cmd_thr) & (wz < cref)
    s = cmd_relative_std(std, wz, cref, fmin)
    return torch.where(narrow, torch.exp(-ang_vel_error / torch.square(s)), base)
