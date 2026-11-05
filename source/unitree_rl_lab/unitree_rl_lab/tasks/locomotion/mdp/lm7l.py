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
