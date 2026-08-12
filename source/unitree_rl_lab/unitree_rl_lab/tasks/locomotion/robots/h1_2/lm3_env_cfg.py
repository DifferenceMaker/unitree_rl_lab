"""LM3 — commanded-velocity walking on the p13c ECONOMY (from scratch).

Operator design 2026-08-12. The verdict from the lm2-vs-p13c comparison: the
two are economic opposites (lm2 earns by MOVING and cannot stand still; p13c
earns by STANDING and steps once per 44 s). LM3 = p13c's push-hardened
stillness economy with commanded velocity strapped on — 27 actions, from
scratch (13->27 action growth makes the weights untransplantable; the pedigree
carried over is the DESIGN).

What changes vs the p13c trunk (balance_env_cfg.RobotEnvCfg):
  * ACTIONS: all 27 joints (arms included). Option II is off — the arm command
    stops driving the arms (apply_directly=False) and becomes a TARGET the
    policy is paid to track (joint_target_deviation_l1), resampled every 10 s,
    wobble off, amplitudes halved (operator: mostly-calm arm work).
  * COMMANDS: base_velocity un-parked (was rel_standing_envs=1.0, ranges 0):
    vx [-0.3, 1.0], vy +-0.3, wz +-0.5, resample 3-10 s, 25% pure standing —
    sudden stops included by construction.
  * REMOVED (the anti-walking genes): feet_air_time_step (engineered to make
    steps COST), foot_displacement_l2_from_spawn, heading_l2_from_spawn,
    heading_stable_bonus (heading is commanded now; wz tracking owns it).
  * GATED ON STANDING (|cmd|<0.1): base_pos_xy hold (with reanchor_on_stop:
    the anchor re-captures WHERE THE COMMAND DROPPED TO ZERO — walk, stop,
    and the p13c station-keeping re-engages at the new spot), torso_lin_vel_xy
    and torso_ang_vel (both WORLD-frame — ungated they tax walking itself).
  * RAISED: track_lin_vel_xy 1.5->3.0, track_ang_vel_z 0.75->1.5 (tracking is
    the ONLY pressure to walk against alive=30), feet_slide -1->-2,
    + feet_clearance 12 (the p13b_clearance gene, swing-height discipline).
  * ADDED: feet_gait_recovery 0.5 gated on speed (phasing income only while
    moving — raise in a later phase, not at the get-go).
  * KEPT: alive 30, the full push ladder + curricula (push_velocity,
    sustained_push), dof_pos_limits -20, stance terms, DR set.
"""
import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg, BODY_JOINT_REGEX
from .balance_env_cfg_queue import _apply_overrides, _load_overrides


def _make_lm3(cfg):
    # ---- commands: un-park the velocity command ----
    c = cfg.commands.base_velocity
    c.resampling_time_range = (3.0, 10.0)
    c.rel_standing_envs = 0.25
    c.ranges.lin_vel_x = (-0.3, 1.0)
    c.ranges.lin_vel_y = (-0.3, 0.3)
    c.ranges.ang_vel_z = (-0.5, 0.5)
    c.limit_ranges.lin_vel_x = (-0.3, 1.0)
    c.limit_ranges.lin_vel_y = (-0.3, 0.3)
    c.limit_ranges.ang_vel_z = (-0.5, 0.5)

    # ---- arms: policy-owned; command becomes a slow, calm target ----
    a = cfg.commands.arm_pose_command
    a.apply_directly = False
    a.resampling_time_range = (10.0, 10.0)
    cfg.curriculum.arm_pose.params["wobble_amplitude_levels"] = (0.0, 0.0, 0.0, 0.0)
    cfg.curriculum.arm_pose.params["pitch_amplitude"] = 0.75
    cfg.curriculum.arm_pose.params["roll_amplitude"] = 0.5
    cfg.curriculum.arm_pose.params["elbow_amplitude"] = 0.75

    # ---- actions: whole body ----
    cfg.actions.JointPositionAction.joint_names = BODY_JOINT_REGEX

    # ---- observations: + velocity command (policy and critic) ----
    for grp in (cfg.observations.policy, cfg.observations.critic):
        grp.velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

    # ---- rewards ----
    r = cfg.rewards
    r.track_lin_vel_xy.weight = 3.0
    r.track_ang_vel_z.weight = 1.5
    r.feet_slide.weight = -2.0
    # the anti-walking genes, out
    r.feet_air_time_step = None
    r.foot_displacement_l2_from_spawn = None
    r.heading_l2_from_spawn = None
    r.heading_stable_bonus = None
    # standing-gated replacements (world-frame terms tax walking ungated)
    r.base_pos_xy_l2_from_spawn = RewTerm(
        func=mdp.base_pos_xy_hold_standing, weight=-0.75,
        params={"command_name": "base_velocity"},
    )
    r.torso_lin_vel_xy = RewTerm(
        func=mdp.body_lin_vel_xy_l2_standing, weight=-3.0,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
    )
    r.torso_ang_vel = RewTerm(
        func=mdp.body_ang_vel_l2_standing, weight=-2.0,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])},
    )
    # swing-height discipline (the p13b_clearance gene, raised)
    r.feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward, weight=12.0,
        params={"std": 0.05, "tanh_mult": 2.0, "target_height": 0.15,
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])},
    )
    # phasing income, speed-gated (raise later, not at the get-go)
    r.gait = RewTerm(
        func=mdp.feet_gait_recovery, weight=0.5,
        params={"period": 0.75, "offset": [0.0, 0.5], "threshold": 0.55,
                "gate_speed": 0.15,
                "sensor_cfg": SceneEntityCfg("contact_forces",
                                             body_names=[".*_ankle_roll_link"])},
    )
    # arm discipline: track the commanded pose (policy owns the arms now)
    r.arm_cmd_track = RewTerm(
        func=mdp.joint_target_deviation_l1, weight=-0.5,
        params={"command_name": "arm_pose_command",
                "asset_cfg": SceneEntityCfg("robot",
                                            joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
                                            preserve_order=True)},
    )

    # ---- events: re-anchor the hold at every stop ----
    cfg.events.reanchor_on_stop = EventTerm(
        func=mdp.reanchor_on_stop, mode="interval", interval_range_s=(0.02, 0.02),
        params={"command_name": "base_velocity", "threshold": 0.1},
    )


@configclass
class RobotEnvCfgLM3(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_lm3(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgLM3(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_lm3(self)
        self.commands.base_velocity.debug_vis = True
        _apply_overrides(self, _load_overrides())
