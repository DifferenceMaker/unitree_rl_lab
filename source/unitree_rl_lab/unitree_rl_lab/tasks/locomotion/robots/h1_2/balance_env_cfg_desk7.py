"""Desk7 — the HIP-HINGE lean trunk (dp7_hinge promoted, operator 2026-09-03:
"promote the hinge as the parent and trunk, so that every job is just deltas.
I think hinge was the winning recipe here and will be used").

Desk6b + the dp7_hinge stack baked (values 1:1 from the harvested
dp7_hinge_2026-09-02 milestone env.yaml):

  * joint_deviation_hips -> joint_deviation_l1_lean_gated (command_name
    "lean_command", lean_ref 0.35): the hinge IS hip motion — the gate frees
    the hips in proportion to the commanded lean, full price at zero lean.
  * upright_bonus (lean_track_bonus) 1.0/0.05 -> 8.0/0.1: the snap tracker
    (dp6c_leancomp temperament — transit time forfeits up to 8/step).
  * base_height -> weight 0.0: the pinned pelvis height is what forced the
    rigid ankle-pivot; removing it lets the pelvis counterweight BACK when
    the torso pitches forward (the human bow). The dp6c xy compensation is
    ABSENT by construction — Desk6b's plain trunk anchors are hinge-compatible
    (pelvis over the feet).

Sim2sim 2026-09-03: "the leaning mechanism of pelvis vs lean works wonders.
It really did learn to pull its pelvis back when leaning more frontally...
the leaning work is by far the best. Not even a compition." Same 91-obs
contract as Desk6/Desk6b — dp7_hinge warmstarts stay valid.
"""
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .balance_env_cfg_desk5b import _make_desk, _make_desk5, _make_desk5b
from .balance_env_cfg_desk6 import _make_desk6, _make_desk6b


def _make_desk7(cfg):
    # the dp7_hinge stack, baked
    cfg.rewards.joint_deviation_hips.func = mdp.joint_deviation_l1_lean_gated
    cfg.rewards.joint_deviation_hips.params["command_name"] = "lean_command"
    cfg.rewards.joint_deviation_hips.params["lean_ref"] = 0.35
    cfg.rewards.upright_bonus.weight = 8.0
    cfg.rewards.upright_bonus.params["std"] = 0.1
    cfg.rewards.base_height.weight = 0.0


@configclass
class RobotEnvCfgDesk7(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_desk(self)
        _make_desk5(self)
        _make_desk5b(self)
        _make_desk6(self)
        _make_desk6b(self)
        _make_desk7(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgDesk7(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_desk(self)
        _make_desk5(self)
        _make_desk5b(self)
        _make_desk6(self)
        _make_desk6b(self)
        _make_desk7(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
