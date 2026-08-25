"""LM5-FSB — FastSAC Run B: the LM5 world with holosoma's minimalist rewards.

Run A (fastsac_lm5_pilotA) = FastSAC x our 30-term ledger — isolates the
ALGORITHM. This cfg is Run B = FastSAC x the paper's 10-term reward set
(g1_29dof_loco_fast_sac preset, weights verbatim) — isolates the REWARDS.
Everything else stays LM5: scene, obs, actions, commands, terminations, DR,
push/velocity curricula (the velocity gates read track_lin_vel_xy /
track_ang_vel_z, and those names are kept).

Term mapping (holosoma -> here):
  tracking_lin_vel 2.0 (exp(-err/0.25))  -> track_lin_vel_xy, std=sqrt(0.25)
  tracking_ang_vel 1.5                   -> track_ang_vel_z,  std=sqrt(0.25)
  penalty_ang_vel_xy -1.0                -> mdp.ang_vel_xy_l2
  penalty_orientation -10.0              -> mdp.flat_orientation_l2
  penalty_action_rate -2.0               -> mdp.action_rate_l2
  feet_phase 5.0 (0.09 / 0.008)          -> mdp.feet_phase (holosoma_rewards)
  pose -0.5 (per-joint weights)          -> mdp.pose_deviation_weighted
  penalty_close_feet_xy -10.0 (0.15)     -> mdp.close_feet_lateral
  penalty_feet_ori -5.0                  -> mdp.feet_flat_orientation
  alive 10.0                             -> mdp.is_alive

Their G1 pose weights by joint role (their 29-dof list decoded): hip_pitch
0.01, hip_roll 1.0, hip_yaw 5.0, knee 0.01, ankles 5.0, waist+arms 50.0 —
mapped to H1-2's 27 dof by name below. Their penalty_curriculum ramp is NOT in
FSB (weights at full strength from step 0) — FSB2 below adds it verbatim.
"""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .lm5_env_cfg import RobotEnvCfgLM5, RobotPlayEnvCfgLM5

FOOT = [".*_ankle_roll_link"]

# H1-2 joint-role weights translated from the G1 preset's pose_weights list
POSE_WEIGHTS = {
    ".*_hip_pitch_joint": 0.01,
    ".*_hip_roll_joint": 1.0,
    ".*_hip_yaw_joint": 5.0,
    ".*_knee_joint": 0.01,
    ".*_ankle_pitch_joint": 5.0,
    ".*_ankle_roll_joint": 5.0,
    "torso_joint": 50.0,
    ".*_shoulder_pitch_joint": 50.0,
    ".*_shoulder_roll_joint": 50.0,
    ".*_shoulder_yaw_joint": 50.0,
    # H1-2 arm naming: elbow_pitch/elbow_roll (roll plays wrist_roll's part),
    # wrist has only pitch/yaw
    ".*_elbow_pitch_joint": 50.0,
    ".*_elbow_roll_joint": 50.0,
    ".*_wrist_pitch_joint": 50.0,
    ".*_wrist_yaw_joint": 50.0,
    # finger joints exist in the articulation but are not policy-actuated —
    # zero weight (holosoma's G1 had no hands in the pose vector)
    "L_.*": 0.0,
    "R_.*": 0.0,
}


def _make_fsb_rewards(cfg):
    # wipe the entire LM5 ledger
    for name in list(vars(cfg.rewards).keys()):
        setattr(cfg.rewards, name, None)

    # tracking pair — names kept so lin/ang_vel_levels curriculum gates work
    cfg.rewards.track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    cfg.rewards.track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    cfg.rewards.ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.0)
    cfg.rewards.flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-10.0)
    cfg.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2.0)
    cfg.rewards.feet_phase = RewTerm(
        func=mdp.feet_phase, weight=5.0,
        params={"command_name": "base_velocity", "std": 0.008, "swing_height": 0.09,
                "period": 1.0, "period_rand_width": 0.2, "foot_rest_z": 0.047,  # measured: ankle_roll z at stand (comx06 keyframe)
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT)},
    )
    cfg.rewards.pose = RewTerm(
        func=mdp.pose_deviation_weighted, weight=-0.5,
        params={"weights": POSE_WEIGHTS},
    )
    cfg.rewards.close_feet = RewTerm(
        func=mdp.close_feet_lateral, weight=-10.0,
        params={"close_feet_threshold": 0.15,
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT)},
    )
    cfg.rewards.feet_ori = RewTerm(
        func=mdp.feet_flat_orientation, weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT)},
    )
    cfg.rewards.alive = RewTerm(func=mdp.is_alive, weight=10.0)


@configclass
class RobotEnvCfgLM5FSB(RobotEnvCfgLM5):
    def __post_init__(self):
        super().__post_init__()
        _make_fsb_rewards(self)


@configclass
class RobotPlayEnvCfgLM5FSB(RobotPlayEnvCfgLM5):
    def __post_init__(self):
        super().__post_init__()
        _make_fsb_rewards(self)


# ── FSB2: + their penalty curriculum (the one recipe piece v1 skipped) ─────────
# holosoma g1_29dof_curriculum_fast_sac, params verbatim: penalties tagged
# penalty_curriculum start at 0.5x and ramp to 1.0x as the running average
# episode length climbs past 750 steps (down toward 0.5x while < 150).
# Pilot B (FSB, full-strength penalties from step 0) plateaued at ~460-step
# survival by step 5k while pilot A (our ledger) broke through — the ramp is
# the first-order suspect, so B2 isolates exactly that.
PENALTY_TERMS = ["ang_vel_xy", "flat_orientation", "action_rate", "pose", "close_feet", "feet_ori"]


def _make_fsb2(cfg):
    from isaaclab.managers import CurriculumTermCfg as CurrTerm

    cfg.curriculum.penalty_curriculum = CurrTerm(
        func=mdp.penalty_curriculum,
        params={"term_names": PENALTY_TERMS, "initial_scale": 0.5, "min_scale": 0.5,
                "max_scale": 1.0, "level_down_threshold": 150.0,
                "level_up_threshold": 750.0, "degree": 0.001,
                "num_compute_average_epl": 1000},
    )


@configclass
class RobotEnvCfgLM5FSB2(RobotEnvCfgLM5FSB):
    def __post_init__(self):
        super().__post_init__()
        _make_fsb2(self)


@configclass
class RobotPlayEnvCfgLM5FSB2(RobotPlayEnvCfgLM5FSB):
    def __post_init__(self):
        super().__post_init__()
        _make_fsb2(self)
