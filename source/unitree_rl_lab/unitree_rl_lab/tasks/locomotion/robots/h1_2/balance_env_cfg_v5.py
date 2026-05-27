"""Phase 5 v5 — three reward-redesign variants tested simultaneously.

All variants inherit v3/v4 EventCfg + CurriculumCfg + CommandsCfg unchanged:
- Mass DR (±3/+5 kg), Friction DR (0.7-1.3)
- Motor strength DR (±15% kp, ±10% kd, per-episode)
- Per-episode push sampling from [0, max]
- capture_spawn_state event for from_spawn reward terms
- Wobble pinned at 0.15

Only RewardsCfg differs between A/B/C.
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from .balance_env_cfg import (
    RobotEnvCfg,
    RobotPlayEnvCfg,
    LEGS_TORSO_JOINT_REGEX,
)
import math


# =============================================================================
# Variant A — PAIRING: every drift penalty paired with positive companion
# =============================================================================
@configclass
class RewardsCfgV5A:
    """17 terms total: 9 positive + 8 negative.
    
    Philosophy: penalty teaches "out of bounds bad", positive teaches
    "in this region good". Both gradients available for PPO.
    """
    # --- Positive (9 terms) ---
    alive = RewTerm(func=mdp.is_alive, weight=20.0)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,  # bumped from v4's 1.0
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,  # bumped from v4's 0.5
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    foot_stance_tracking = RewTerm(
        func=mdp.foot_stance_tracking,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "std": 0.08,
            "nominal_foot_pos_b": [[0.0, 0.10], [0.0, -0.10]],
        },
    )
    torso_stability_bonus = RewTerm(
        func=mdp.torso_stability_bonus,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "std_lin": 0.10,
            "std_ang": 0.30,
        },
    )
    heading_stable_bonus = RewTerm(func=mdp.heading_stable_bonus, weight=1.5)
    centered_pose_bonus = RewTerm(func=mdp.centered_pose_bonus, weight=1.0)
    upright_bonus = RewTerm(func=mdp.upright_bonus, weight=1.0)

    # --- Negative (8 terms) ---
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEGS_TORSO_JOINT_REGEX)},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 1.0})
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["torso_link", ".*hip.*", ".*knee.*"])},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    heading_l2_from_spawn = RewTerm(func=mdp.heading_l2_from_spawn, weight=-2.0)
    base_pos_xy_l2_from_spawn = RewTerm(func=mdp.base_pos_xy_l2_from_spawn, weight=-1.0)
    foot_displacement_l2_from_spawn = RewTerm(
        func=mdp.foot_displacement_l2_from_spawn,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])},
    )
    joint_deviation_legs_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["torso_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )


# =============================================================================
# Variant B — REPLACING: positive-only shaping, penalties only for safety
# =============================================================================
@configclass
class RewardsCfgV5B:
    """12 terms total: 9 positive + 3 negative.
    
    Philosophy: positive shaping rewards push policy toward correct behavior
    via gradient toward exp peak. L2 penalties only for hard safety bounds
    (joint limits, height, undesired contacts).
    """
    # --- Positive (9 terms) ---
    alive = RewTerm(func=mdp.is_alive, weight=20.0)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    foot_stance_tracking = RewTerm(
        func=mdp.foot_stance_tracking,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "std": 0.08,
            "nominal_foot_pos_b": [[0.0, 0.10], [0.0, -0.10]],
        },
    )
    torso_stability_bonus = RewTerm(
        func=mdp.torso_stability_bonus,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "std_lin": 0.10,
            "std_ang": 0.30,
        },
    )
    heading_stable_bonus = RewTerm(func=mdp.heading_stable_bonus, weight=2.0)  # higher — no penalty
    centered_pose_bonus = RewTerm(func=mdp.centered_pose_bonus, weight=1.5)
    foot_planted_bonus = RewTerm(
        func=mdp.foot_planted_bonus,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "std": 0.05,
        },
    )
    upright_bonus = RewTerm(func=mdp.upright_bonus, weight=1.5)

    # --- Negative (3 terms — SAFETY ONLY) ---
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEGS_TORSO_JOINT_REGEX)},
    )
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 1.0})
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["torso_link", ".*hip.*", ".*knee.*"])},
    )


# =============================================================================
# Variant C — SPARSE PAIRING: minimal terms, only critical pairings
# =============================================================================
@configclass
class RewardsCfgV5C:
    """12 terms total: 7 positive + 5 negative.
    
    Philosophy: keep only the strictly necessary terms. Drop foot_displacement
    entirely (trust foot_stance_tracking). Drop base_pos_xy positive (penalty
    enough). Mid-ground between A and B.
    """
    # --- Positive (7 terms) ---
    alive = RewTerm(func=mdp.is_alive, weight=20.0)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    foot_stance_tracking = RewTerm(
        func=mdp.foot_stance_tracking,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "std": 0.08,
            "nominal_foot_pos_b": [[0.0, 0.10], [0.0, -0.10]],
        },
    )
    torso_stability_bonus = RewTerm(
        func=mdp.torso_stability_bonus,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "std_lin": 0.10,
            "std_ang": 0.30,
        },
    )
    heading_stable_bonus = RewTerm(func=mdp.heading_stable_bonus, weight=1.5)
    upright_bonus = RewTerm(func=mdp.upright_bonus, weight=1.0)

    # --- Negative (5 terms) ---
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEGS_TORSO_JOINT_REGEX)},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 1.0})
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["torso_link", ".*hip.*", ".*knee.*"])},
    )
    heading_l2_from_spawn = RewTerm(func=mdp.heading_l2_from_spawn, weight=-2.0)


# =============================================================================
# Three Env subclasses — each pulls its own RewardsCfg variant
# =============================================================================
@configclass
class RobotEnvCfgV5A(RobotEnvCfg):
    rewards: RewardsCfgV5A = RewardsCfgV5A()


@configclass
class RobotEnvCfgV5B(RobotEnvCfg):
    rewards: RewardsCfgV5B = RewardsCfgV5B()


@configclass
class RobotEnvCfgV5C(RobotEnvCfg):
    rewards: RewardsCfgV5C = RewardsCfgV5C()


# Optional play env subclasses for visual evaluation
@configclass
class RobotPlayEnvCfgV5A(RobotPlayEnvCfg):
    rewards: RewardsCfgV5A = RewardsCfgV5A()

@configclass
class RobotPlayEnvCfgV5B(RobotPlayEnvCfg):
    rewards: RewardsCfgV5B = RewardsCfgV5B()

@configclass
class RobotPlayEnvCfgV5C(RobotPlayEnvCfg):
    rewards: RewardsCfgV5C = RewardsCfgV5C()