"""HipsArm -- the p14g_hips_armfull recipe promoted to TRUNK (operator 2026-09-28, after the p14g
hardware round: "hips_armfull is a safe parent and a good one. Make it the default" / "Make
hips_armfull the new balance default and a new trunk, with hips -1.0 and the full arm envelope
baked, so the next rows are pure deltas -- Yes I agree 100%").

WHY IT IS THE TRUNK. Hardware 2026-09-28, six of six p14g rows passed with zero near-zone steps;
hips_armfull was the operator's best of all: the IK resolver's accidental over-the-top "dunk" arm
motion produced NO recovery step ("The arm curric wholly worked ... really profound"), and it rests
the most upright of the round (0.43 deg back, roll 0.09). The 2026-09-17 extreme-arm fall class is
closed on this recipe.

Stance trunk (kitchen + feet_too_near 0.25, real-hand body) + EXACTLY what the p14g_hips_armfull
job applied on top of p14f_stance_on_tilt, transcribed 1:1 from scripts/queue/p14g/p14g_hips_armfull.job
and verified against milestone_checkpoints/p14g_hips_armfull_2026-09-23/params/env.yaml:
  * tilt: upright_bonus std 0.05, w 3.0 (the LOCKED lean lever; the Stance trunk does not carry it,
    every p14f/p14g row re-applied it -- terms do not travel with a checkpoint)
  * hips: joint_deviation_hips -0.225 -> -1.0
  * the full arm envelope: arm_pose_command resampling (2,10) -> (1,5) s, max_joint_speed 0.6 -> 1.2,
    orientation_delta_rpy (2.4,.8,.8) -> (3.14,1.5,1.5), orientation_prob .7 -> .9
Parent for warmstarts: milestones/p14g_hips_armfull_2026-09-23 (model_75600, +5651 past
p14f_stance_on_tilt@69950; the run died at ~+6765 so the pick is "just before the dip" -- a fine
checkpoint to deploy, and the stack procedure is the only warmstart recipe that has survived).

Contract unchanged from Stance: same 87 obs, QueueFixedLRPPORunnerCfg (fixed lr 3e-4, GuardedPPO),
same experiment_name. Nothing else in this file is a delta -- deltas live in the jobs.
"""

from isaaclab.utils import configclass

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_kitchen import _use_real_hand
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .balance_env_cfg_queue_ik import _swap_in_ik_command
from .balance_env_cfg_stance import _make_stance

HIPSARM_UPRIGHT_STD = 0.05
HIPSARM_UPRIGHT_W = 3.0
HIPSARM_HIPS_W = -1.0
HIPSARM_RESAMPLING = (1.0, 5.0)
HIPSARM_MAX_JOINT_SPEED = 1.2
HIPSARM_ORIENTATION_DELTA_RPY = (3.14, 1.5, 1.5)
HIPSARM_ORIENTATION_PROB = 0.9


def _make_hipsarm(cfg):
    """Stance, then the p14g_hips_armfull edits. Order matters: stance sets the base first."""
    _make_stance(cfg)
    R = cfg.rewards
    R.upright_bonus.params["std"] = HIPSARM_UPRIGHT_STD
    R.upright_bonus.weight = HIPSARM_UPRIGHT_W
    R.joint_deviation_hips.weight = HIPSARM_HIPS_W
    C = cfg.commands.arm_pose_command
    C.resampling_time_range = HIPSARM_RESAMPLING
    C.max_joint_speed = HIPSARM_MAX_JOINT_SPEED
    C.orientation_delta_rpy = HIPSARM_ORIENTATION_DELTA_RPY
    C.orientation_prob = HIPSARM_ORIENTATION_PROB
    assert R.upright_bonus.params["std"] == 0.05 and R.upright_bonus.weight == 3.0, "[HipsArm trunk] tilt must be std .05 w3"
    assert R.joint_deviation_hips.weight == -1.0, "[HipsArm trunk] hips must be -1.0"
    assert R.feet_too_near.params["threshold"] == 0.25, "[HipsArm trunk] the Stance threshold must survive"
    assert tuple(C.resampling_time_range) == (1.0, 5.0) and C.max_joint_speed == 1.2, "[HipsArm trunk] arm timing/rate"
    assert tuple(C.orientation_delta_rpy) == (3.14, 1.5, 1.5) and C.orientation_prob == 0.9, "[HipsArm trunk] arm rotation envelope"


@configclass
class RobotEnvCfgHipsArm(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _swap_in_ik_command(self)
        _make_hipsarm(self)
        _use_real_hand(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgHipsArm(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _swap_in_ik_command(self)
        _make_hipsarm(self)
        _use_real_hand(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
