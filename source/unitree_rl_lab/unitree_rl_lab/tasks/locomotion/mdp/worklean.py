"""WorkLean terms (dp9d, 2026-10-09). Own module on purpose: the pod's mdp/__init__.py differs from this checkout's
(it does not import holosoma_rewards), so new terms arrive as a new file + one appended import line, never as a copied
__init__.py or an edited rewards.py."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height_l2_lean_gated(
    env: "ManagerBasedRLEnv",
    target_height: float,
    command_name: str = "lean_command",
    lean_ref: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """dp9d crouch (operator 2026-10-09: "I sort of do like the sitting stance but i don't want it to conflict with
    leaning. Can the base_height be set at the crouch with the weight but gated on the lean command?"): base_height_l2 at a
    FIXED crouch target, faded with the commanded |lean| exactly as joint_deviation_l1_lean_gated fades the hip price --
    full price at cmd 0, zero at |cmd| >= lean_ref. base_height_lean_l2 instead keeps the price on at every lean and only
    moves the target (target * cos(cmd)), which taxed the hinge's pelvis drop in dp6c (the over-seat). Here nothing pulls on
    the pelvis height while a lean is commanded."""
    asset = env.scene[asset_cfg.name]
    theta = env.command_manager.get_command(command_name)[:, 0].abs()
    fade = 1.0 - (theta / lean_ref).clamp(max=1.0)
    return torch.square(asset.data.root_pos_w[:, 2] - target_height) * fade
