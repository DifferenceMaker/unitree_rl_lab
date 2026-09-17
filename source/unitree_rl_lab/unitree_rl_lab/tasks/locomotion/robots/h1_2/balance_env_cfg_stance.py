"""Stance — the p14b_stance delta promoted to parent + trunk (operator 2026-09-17, after the
hardware session: "SO now stance is inherited and it can learn the delta on top of it?").

Kitchen trunk + ONE thing: `feet_too_near.threshold = 0.25`. That single threshold is the
only change, and it is here rather than in each job for the reason the lineage rule exists —
terms do NOT travel with a checkpoint, so a p14c job warmstarted from p14b_stance that did
not RESTATE 0.25 would silently revert to kitchen's 0.22 and un-fix the creep on iteration 1.
Baking it makes every p14c row a pure delta on exactly what stance trained on, the same
contract kitchen gave p14b.

WHY IT IS THE TRUNK. Hardware 2026-09-17, operator: "The stance issue of the left leg being
inward is real. Stance solved it." ... "Stance had the best stance." One threshold closed a
defect that had survived a leg swap, a body change and four waves of reward work. The
mechanism, corrected from the harvest note that called this term inert: feet_too_near is
(threshold - d) clamped at 0 at weight -8, so its INCOME is the residual violation, not the
pressure applied. plant earns -0.002 at 0.22 (0.25 mm mean violation, feet ride at ~0.22);
stance earns -0.006 at 0.25 (0.75 mm, feet ride at ~0.25). Had stance kept plant's separation
its income would have been -8 x 0.03 = -0.24, forty times what it shows. The feet moved out
the full 3 cm. A quiet hinge is a hinge being obeyed.

BODY: the trunk keeps the real-hand plant (h1_2_comx06_hand790.urdf) via _use_real_hand.
A job may swap it with `set_body <basename>` for a PLANT probe (the p14c comy rows); the
swap is asserted and printed, never silent.

Contract unchanged from Kitchen: same 87 obs, same QueueFixedLRPPORunnerCfg (fixed lr 3e-4,
GuardedPPO), same experiment_name, so p14b_stance warmstarts and harvest paths just work.
Nothing else in this file is a delta - deltas live in the jobs.
"""

from isaaclab.utils import configclass

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_kitchen import _make_kitchen, _use_real_hand
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .balance_env_cfg_queue_ik import _swap_in_ik_command

# The p14b_stance delta, as the job stated it.
STANCE_FEET_TOO_NEAR_THRESHOLD = 0.25


def _make_stance(cfg):
    """Kitchen, then the one p14b_stance edit. Order matters: kitchen sets 0.22 first."""
    _make_kitchen(cfg)
    cfg.rewards.feet_too_near.params["threshold"] = STANCE_FEET_TOO_NEAR_THRESHOLD
    assert cfg.rewards.feet_too_near.params["threshold"] == 0.25, (
        "[Stance trunk] feet_too_near.threshold must be 0.25 - it is the whole point of this trunk")


@configclass
class RobotEnvCfgStance(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _swap_in_ik_command(self)
        _make_stance(self)
        _use_real_hand(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgStance(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _swap_in_ik_command(self)
        _make_stance(self)
        _use_real_hand(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
