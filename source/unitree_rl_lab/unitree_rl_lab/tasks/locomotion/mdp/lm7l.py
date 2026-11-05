"""lm7l (2026-11-05): the leg-roll ENCODER-OFFSET DR on the walk line (operator, after both greenlit lm7k rows walked with a RIGHT lean on the
robot: "I think (from my experience) the answer lays in one of these in priority: 1. CoM. 2. Some encoder error." -> "Actually might be worth
it. Yes lets test this. Author and smoke it.").

The p16_dr mechanism (mdp/dr.py) unchanged: `randomize_encoder_offset` (reset event) draws a constant per-joint offset per episode, the policy
observes joint_pos_rel + offset (`joint_pos_rel_encoder`), the applied target is (raw x scale + default) - offset (the PD runs in the encoder
frame). The walk's action term is `ArmGravityFFJointPositionAction` (lm7j: the arm gravity feed-forward), so the subtraction lives in a
subclass of THAT term here; while the offset buffer is zero it is bit-identical to the parent's term. The critic keeps the true joint_pos_rel
(same width -> warmstart-compatible; p16_dr's enc03/enc05 on 13 joints spiked the critic -- this row opens the band on the 4 ROLL joints only).
"""
from __future__ import annotations

import dataclasses

import torch
from isaaclab.utils import configclass

from .lm7j import ArmGravityFFJointPositionAction, ArmGravityFFJointPositionActionCfg


class ArmGravityFFEncJointPositionAction(ArmGravityFFJointPositionAction):
    """ArmGravityFFJointPositionAction whose applied target is (raw * scale + default) - env.encoder_offset (zeros until an event fills it)."""

    cfg: "ArmGravityFFEncJointPositionActionCfg"

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        enc = getattr(self._env, "encoder_offset", None)
        if enc is not None:
            self._processed_actions = self._processed_actions - enc[:, self._joint_ids]


@configclass
class ArmGravityFFEncJointPositionActionCfg(ArmGravityFFJointPositionActionCfg):
    class_type: type = ArmGravityFFEncJointPositionAction


def arm_gravity_ff_enc_action_cfg(a: ArmGravityFFJointPositionActionCfg) -> ArmGravityFFEncJointPositionActionCfg:
    """The Enc cfg carrying every field of the existing ArmGravityFFJointPositionActionCfg `a` (scale, offset, clip, joints, ff settings)."""
    kw = {f.name: getattr(a, f.name) for f in dataclasses.fields(a) if f.name != "class_type"}
    return ArmGravityFFEncJointPositionActionCfg(**kw)


# ---------------------------------------------------------------------------------------------------------------------------------
# lm7l_legsout (2026-11-05 evening, operator: "What we need to kill is the inward leg movement ... I really want the inward leg movement
# to be gone, alongside armsway and leaning"): the parent's WALKING ledger has two width attractors that disagree -- foot_stance_tracking
# (w 3, exp(-sum|foot - nominal| / .08)) pulls the feet to nominal_walk y +-0.10 = a 20 cm width, feet_width_walking (w 1.5, std .06)
# pulls to 30 cm; summed, the ledger's optimum under a command is 20 cm (3.09/step) with a near-flat shelf 24-30 cm (2.37/step) -- the
# inward walk IS the ledger's optimum (the Bible's lesson), and the only wall, feet_too_near, is a hinge on the PLANAR distance, which
# the fore-aft stagger inflates mid-stride so the lateral width can shrink under it unpriced. The gene aligns the nominal (y +-0.15)
# and adds the wall below on the LATERAL width itself.
# ---------------------------------------------------------------------------------------------------------------------------------
from isaaclab.managers import SceneEntityCfg
from .events import feet_lateral_sep_b
from .lm7j import _walking


def feet_lateral_width_hinge_walking(env, asset_cfg: SceneEntityCfg, w_min: float = 0.26, command_name: str = "base_velocity",
                                     cmd_thr: float = 0.03) -> torch.Tensor:
    """Hinge (w_min - w)+ on the signed LATERAL foot separation in the base frame (left y - right y) while |cmd| > cmd_thr. A wall, not a
    kernel: zero at and above w_min, linear below (no flat region where the robot lives; a crossed stance, w < 0, pays w_min + |w|).
    NEGATIVE weight. At -15: 2 cm inside the wall costs 0.3 / step, 6 cm (the 20 cm walk) 0.9 / step -- above the 0.7 / step the
    20 cm optimum earns over the 30 cm one in the parent's two width kernels. Earns ~0 once the walk stays at >= 26 cm (rule: a
    working wall's income is the residual violation)."""
    w = feet_lateral_sep_b(env, asset_cfg)
    return (w_min - w).clamp(min=0.0) * _walking(env, command_name, cmd_thr).float()
