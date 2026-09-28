"""LM6 reward variants (2026-09-28): the p14 balance-phase kernels ported to locomotion
WITH the locomotion gating principle kept (operator: "remember to leave the gating
principle on locomotion still").

Both gates read the COMMAND, which is in the policy observation (velocity_commands),
never the base velocity -- p14_recovery_gate (2026-09-15) proved a gate on state the
policy cannot see is reward noise, not a gate. Gated PENALTIES and gated INCOME that
is only meaningful in the gated state are both legal (Bible law 1 forbids gated income
next to UNGATED penalties on the same behaviour).
"""

import torch
from isaaclab.managers import SceneEntityCfg

from .rewards import feet_crossed, upright_bonus


def upright_bonus_standing(
    env,
    std: float = 0.05,
    command_name: str = "base_velocity",
    thr: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The p14 lean lever (upright_bonus std .05 w3, LOCKED on the balance line 2026-09-23)
    paid only while the command says STAND (|cmd| < thr). A walking torso must pitch:
    the LM5 walker holds ~5 deg pitch rms, where a std .05 kernel earns <3% and is as
    dead as the std .01 one it replaces (2% earned, lm5h ledger 2026-09-28). While
    walking, flat_orientation_l2 (-4.5) stays the only orientation price."""
    base = upright_bonus(env, std=std, asset_cfg=asset_cfg)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    return base * (cmd < thr).float()


def feet_crossed_cmd_gated(
    env,
    min_gap: float = 0.05,
    command_name: str = "base_velocity",
    vy_max: float = 0.1,
    wz_max: float = 0.2,
    left_body: str = "left_ankle_roll_link",
    right_body: str = "right_ankle_roll_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """p13g/p14 recovery-geometry hinge (feet_crossed -8, min_gap .05) made walking-safe:
    OFF while a sideways (|vy_cmd| >= vy_max) or turning (|wz_cmd| >= wz_max) command is
    active, because a crossover step is HOW a biped walks sideways and turns. Forward /
    backward walking and standing keep the hinge: a crossed stance there is the fall type
    the term was written for."""
    base = feet_crossed(env, min_gap=min_gap, left_body=left_body, right_body=right_body, asset_cfg=asset_cfg)
    cmd = env.command_manager.get_command(command_name)
    gate = (cmd[:, 1].abs() < vy_max) & (cmd[:, 2].abs() < wz_max)
    return base * gate.float()
