"""LM5 — the lm4e_wall_hips recipe TRUNKED. Operator ruling 2026-08-20:
"wall_hips is the new parent. Trunk it. MAKE SURE TO INCLUDE ALL THE REWARDS
AND CURRICULUMS."

Why wall_hips: the ONLY policy of the lm4e wave that UNPARKED left_hip_yaw
(probe 2026-08-20: limit occupancy zero across all phases, hips symmetric
-0.01..-0.07 vs the parked -0.430 hard stop; wandb dof_pos_limits -0.088 vs
the -4.33 saturated ceiling on the three parked siblings). First hardware-
legal walk policy on the limits axis (the 2026-07-08 motor-protection
failure mode). Rig verdict: "Excellent stance ... the gait is the best ...
really robust against pushes"; probe gains vx 71-73%, vy 67%, back 53%.
KNOWN OPEN PROBLEMS (lm5 axes, NOT solved here): wz dead (2% of commanded),
heading drift +0.03-0.05 rad/s in every walking phase (no heading loop).

TRUNK-IS-TASK LAW (the p12e inheritance incident + Balance-Q accident): this
file bakes the COMPLETE wall_hips stack — the lm4b trunk (via _make_lm3/
_make_lm4/_make_lm4b: arm-command deletion, cleargate fixes, anti-rocking
pair, command curriculum with gate_frac 0.5) PLUS every job delta from
lm4e_wall_hips's overrides.json, transcribed 1:1 below and smoke-verified by
TABLE DIFF against the milestone's env.yaml. Parent for warmstarts:
milestones/lm4e_wall_hips_2026-08-20 (model_30225).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .lm3_env_cfg import _make_lm3
from .lm4_env_cfg import _make_lm4
from .lm4b_env_cfg import _make_lm4b


def _make_lm5(cfg):
    """The lm4e_wall_hips job deltas, baked (source: its overrides.json)."""
    # ---- crisp tracking economy: weights 10/5, kernels std 0.25, L2-far pair ----
    cfg.rewards.track_lin_vel_xy.weight = 10.0
    cfg.rewards.track_ang_vel_z.weight = 5.0
    cfg.rewards.track_lin_vel_xy.params["std"] = 0.25
    cfg.rewards.track_ang_vel_z.params["std"] = 0.25
    cfg.rewards.track_err = RewTerm(
        func=mdp.track_vel_err_l2, weight=-0.5,
        params={"command_name": "base_velocity"},
    )

    # ---- arms v2: deviation -1.0 INCLUDING shoulder_pitch + swing income ----
    cfg.rewards.joint_deviation_arms.weight = -1.0
    cfg.rewards.joint_deviation_arms.params["asset_cfg"].joint_names = [
        ".*_shoulder_pitch.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*",
        ".*_elbow.*", ".*_wrist.*",
    ]
    cfg.rewards.arm_swing = RewTerm(
        func=mdp.arm_gait_swing, weight=1.0,
        params={"period": 0.75, "offset": [0.5, 0.0], "amplitude": 0.25,
                "std": 0.3, "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_shoulder_pitch_joint"], preserve_order=True)},
    )
    cfg.rewards.standing_pose = RewTerm(
        func=mdp.standing_pose_bonus, weight=2.0,
        params={"std": 0.8, "command_name": "base_velocity"},
    )

    # ---- the unpark pair (lm4e verdict) ----
    # wall: joint_pos_limits saturates at (hard-soft)*w per joint; -100 makes
    # limit-riding economically unthinkable without taxing legitimate motion.
    cfg.rewards.dof_pos_limits.weight = -100.0
    # evictor: pulls a wall-free park (just inside soft) back toward default.
    # NOT higher: hips scope is roll+yaw = the DOFs vy/wz need.
    cfg.rewards.joint_deviation_hips.weight = -2.0


@configclass
class RobotEnvCfgLM5(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_lm3(self)
        _make_lm4(self)
        _make_lm4b(self)
        _make_lm5(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgLM5(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_lm3(self)
        _make_lm4(self)
        _make_lm4b(self)
        _make_lm5(self)
        self.commands.base_velocity.debug_vis = True
        _apply_overrides(self, _load_overrides())
