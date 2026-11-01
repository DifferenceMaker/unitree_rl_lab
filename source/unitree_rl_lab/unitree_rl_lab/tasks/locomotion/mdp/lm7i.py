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


def track_lin_vel_xy_split(
    env: "ManagerBasedRLEnv",
    std: float,
    std_stand: float = 0.5,
    command_name: str = "base_velocity",
    cmd_thr: float = 0.03,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """isaaclab's track_lin_vel_xy_yaw_frame_exp (velocity error in the gravity-aligned yaw frame) with kernel std_stand while the
    full command norm (vx, vy, wz) is below cmd_thr -- the line's standing gate -- and the tracking ladder's std otherwise. At a
    standstill both branches pay the full weight (error 0); after a push the standing branch keeps paying: 20 exp(-0.3^2 / 0.5^2)
    = 13.95 at 0.3 m/s instead of ~0 at std .05."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    cmd = env.command_manager.get_command(command_name)
    err = torch.sum(torch.square(cmd[:, :2] - vel_yaw[:, :2]), dim=1)
    standing = torch.linalg.norm(cmd, dim=1) < cmd_thr
    s = torch.where(standing, torch.full_like(err, std_stand), torch.full_like(err, std))
    return torch.exp(-err / torch.square(s))
