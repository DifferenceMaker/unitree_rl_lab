"""LM4 — lm3 with the arm command channel DELETED, symmetry-aware training,
and a command-magnitude curriculum. Operator design 2026-08-13.

Why (the sim2sim autopsy of lm3_cleargate, 2026-08-13): the trunk's
arm_pose_command obs (14 dims) is fed by a C++ term the walk deploy stack
never publishes — on the rig the policy saw a CONSTANT, deep OOD vs the
sampled holds it trained against. Visible damage: asymmetric arm pose (right
arm raised shoulder-height), arms dropping when velocity commands arrive, and
possibly poisoned velocity tracking. Verdict: the walk line does not carry an
arm command AT ALL. The policy owns all 27 joints; arms are gait actuators
(counter-swing is free), and hands stay at sides by REWARD, not by command.

Changes vs the LM3 trunk (lm3_env_cfg._make_lm3):
  * REMOVED: arm_pose_command (command term, policy+critic obs, arm_pose
    curriculum, arm_cmd_track reward). Actor obs 104 -> 90, critic 134 -> 120.
    SCRATCH ONLY — the obs change breaks any warmstart.
  * ADDED joint_deviation_arms (L1 to defaults, -0.25) on shoulder_roll/yaw +
    elbow + wrist: hands at sides. shoulder_pitch is DELIBERATELY not taxed —
    that is the counter-swing DOF (operator: "they swung their arms counter to
    the opposing leg").
  * BAKED the lm3_cleargate audit fixes (they were job deltas there; trunk-is-
    task says the next trunk carries them): feet_clearance -> command-gated
    variant (no standstill subsidy), base_height target 0.98 (not the desk
    crouch).
  * BAKED the lm3b anti-rocking pair (preview verdict on cleargate: lateral
    rocking while standing): gait 0.5 -> 1.5 (stronger phasing = committed
    weight transfer instead of the metastable both-feet shuffle),
    base_angular_velocity -0.0375 -> -0.3 (roll/pitch rate is the rocking
    motion itself; yaw untouched by ang_vel_xy_l2).
  * COMMAND CURRICULUM (the fwd10 lesson: 52% mean tracking can hide "never
    mastered 1.0 m/s"): ranges START narrow (vx +-[-0.2,0.4], vy +-0.2,
    wz +-0.3) and expand +-0.1 toward the full limits (vx [-0.3,1.0],
    vy +-0.3, wz +-0.5) each time mean tracking reward clears 80% of its
    ceiling (mdp.lin_vel_cmd_levels / ang_vel_cmd_levels, evaluated every
    episode length). The policy only ever trains at commands it is close to
    mastering.

Symmetry training (paper #226) lives in the RUNNER cfg (LM4PPORunnerCfg:
mirror loss via mdp.symmetry:mirror_h1_2_walk), not here. The LCP variant
(paper #14) additionally deletes action_rate — see RobotEnvCfgLM4LCP and
agents/lcp_ppo.py.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .lm3_env_cfg import _make_lm3


def _make_lm4(cfg):
    # ---- the arm command channel, deleted end to end ----
    cfg.commands.arm_pose_command = None
    cfg.curriculum.arm_pose = None
    cfg.rewards.arm_cmd_track = None
    cfg.observations.policy.arm_pose_command = None
    cfg.observations.critic.arm_pose_command = None

    # hands at sides by reward; shoulder_pitch left free for counter-swing
    cfg.rewards.joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.25,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=[".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_elbow.*", ".*_wrist.*"],
        )},
    )

    # ---- cleargate audit fixes, baked ----
    cfg.rewards.feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward_cmd, weight=12.0,
        params={"std": 0.05, "tanh_mult": 2.0, "target_height": 0.15,
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])},
    )
    cfg.rewards.base_height.params["target_height"] = 0.98

    # ---- lm3b anti-rocking pair, baked ----
    cfg.rewards.gait.weight = 1.5
    cfg.rewards.base_angular_velocity.weight = -0.3

    # ---- command-magnitude curriculum ----
    c = cfg.commands.base_velocity
    c.ranges.lin_vel_x = (-0.2, 0.4)
    c.ranges.lin_vel_y = (-0.2, 0.2)
    c.ranges.ang_vel_z = (-0.3, 0.3)
    c.limit_ranges.lin_vel_x = (-0.3, 1.0)
    c.limit_ranges.lin_vel_y = (-0.3, 0.3)
    c.limit_ranges.ang_vel_z = (-0.5, 0.5)
    cfg.curriculum.lin_vel_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels, params={"reward_term_name": "track_lin_vel_xy"}
    )
    cfg.curriculum.ang_vel_levels = CurrTerm(
        func=mdp.ang_vel_cmd_levels, params={"reward_term_name": "track_ang_vel_z"}
    )


@configclass
class RobotEnvCfgLM4(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_lm3(self)
        _make_lm4(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotEnvCfgLM4LCP(RobotEnvCfgLM4):
    """lm4_lcp trunk: lm4 with action_rate DELETED — its job is taken over
    by the LCP gradient penalty in the loss (agents/lcp_ppo.py). An explicit
    cfg class, not a job delta, so env.yaml witnesses it."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate = None


@configclass
class RobotPlayEnvCfgLM4(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_lm3(self)
        _make_lm4(self)
        self.commands.base_velocity.debug_vis = True
        _apply_overrides(self, _load_overrides())
