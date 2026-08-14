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
