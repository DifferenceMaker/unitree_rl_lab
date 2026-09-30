"""Desk8-Work -- the WORK policy trunk (operator 2026-09-30).

ARCHITECTURE RULING. The desk line stops being a walker that also leans. Three policies and one
commander mode: BALANCE (stand anywhere, recover without stepping), WORK (stand at the table, lean,
keep heading, NO transit), WALK (velocity tracking) and HOMING (= WALK driven by a goal-to-velocity
commander back into the work zone). "WORK, displaced beyond a radius, stand into WALK pose, HOMING
until inside the zone with heading, WORK again." So this trunk drops the transit gene's teleports
and freezes the anchor ("the reanchoring will happen via pushing"), and takes the BALANCE line's
robustness recipe instead: its push regime, its arm envelope, its walls and prices.

Built as Desk7 (hinge trunk, 2026-09-03) + the dp8c_yaw job stack transcribed 1:1 from
scripts/queue/dp8c/dp8c_yaw.job (the 93-obs heading contract that survived the 2026-09-30 sim2sim
pass as "the most torso stable, sways the least, keeps heading") + the BALANCE carry list from the
HipsArm trunk (p14 laws), minus the transit pieces:
  dp8c keepers:  feet_slide -5; gait gate prop @0.20; feet_clearance recovery 2.0 gated @0.20;
                 capture_point_touchdown_anchor gate 0.05; feet_air_time_step free_when_displaced 0.05;
                 anchor_hold deadband 0.05; held/dropout anchor obs; clip None; foot_impact_force
                 -0.005; upright (lean tracker) std 0.035 at w 8; torso_stability SPLIT lin 0.15 /
                 ang 0.30 at w 3 each (product term REMOVED); torso_lin_vel_xy -8 (damp);
                 foot_stance_tracking std 0.15; anchor_yaw obs (policy held/dropout, critic clean).
  WORK only:     NO seed_offset teleports, NO move_anchor nudges (Desk5b removed it; dp8c re-added
                 it per job; here it stays off).
  BALANCE carry: action_cap SQUARED -0.05 thr 1.5 (replaces softplus); knee_torque_cap -0.2 @100 Nm;
                 feet_too_far -8 @0.45; feet_crossed -8 gap 0.05; foot_displacement_l2_from_spawn
                 -0.375 (the smooth radial gradient the flat anchor well lacks; 0.002/step inside
                 the 5 cm deadband); joint_vel_reg -0.01; feet_too_near 0.25 (Stance).
                 energy is an AXIS row, not baked.
  base_height:   back at the line's historical weight (-15.0 in 235 milestones) as
                 base_height_lean_l2 target 0.87 -- the target follows the commanded lean, so the
                 hinge is not taxed (the reason Desk7 zeroed the flat version). Operator: "Make sure
                 the base_height term is added and works."
  push regime:   BALANCE's -- push_robot every 8-12 s to 1.5 m/s, sustained 15-25 s to 50 N (the
                 desk line had 20-30 s to 1.0 m/s, 30-50 s to 30 N). "That is where BALANCE's
                 robustness came from and WORK needs the same."
  arm envelope:  HipsArm's -- resampling (1,5) s, max_joint_speed 1.2, orientation_delta_rpy
                 (3.14,1.5,1.5), orientation_prob 0.9 ("the bigger difficulty is really good").
  body:          h1_2_comx06_hand790 (real 790 g hand) + the Unitree armature table.
SCRATCH trunk (no warmstart): "the desk line is now at 69k ... most detonations". Runner
QueueFixedLRPPORunnerCfg (fixed lr 3e-4, GuardedPPO) as every p15 scratch row. 93 obs.
"""
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_desk5b import _make_desk, _make_desk5, _make_desk5b
from .balance_env_cfg_desk6 import _make_desk6, _make_desk6b
from .balance_env_cfg_desk7 import _make_desk7
from .balance_env_cfg_kitchen import _use_real_hand
from .balance_env_cfg_queue import _apply_overrides, _load_overrides

DESK8_BASE_HEIGHT_W = -15.0
DESK8_BASE_HEIGHT_TARGET = 0.87
DESK8_PUSH_LEVELS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
DESK8_SUSTAINED_LEVELS = (
    ((0.0, 0.0), (0.0, 0.0)),
    ((0.0, 15.0), (1.5, 3.0)),
    ((0.0, 30.0), (2.0, 3.5)),
    ((0.0, 40.0), (2.0, 3.5)),
    ((0.0, 50.0), (2.0, 4.0)),
)
DESK8_ARM_RESAMPLING = (1.0, 5.0)
DESK8_ARM_MAX_JOINT_SPEED = 1.2
DESK8_ARM_ORIENTATION_DELTA_RPY = (3.14, 1.5, 1.5)
DESK8_ARM_ORIENTATION_PROB = 0.9
_FEET = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])
_TORSO = SceneEntityCfg("robot", body_names="torso_link")


def _make_desk8_work(cfg):
    R = cfg.rewards
    P = cfg.observations.policy

    # ── 1. the dp8c_yaw stack, 1:1, minus the transit teleports and the anchor nudges ──
    R.feet_slide.weight = -5.0
    R.gait.params["gate_mode"] = "prop"
    R.gait.params["gate_anchor_dist"] = 0.20
    R.feet_clearance = RewTerm(
        func=mdp.foot_clearance_recovery, weight=2.0,
        params={"std": 0.05, "tanh_mult": 2.0, "target_height": 0.12,
                "gate_mode": "prop", "gate_anchor_dist": 0.20, "asset_cfg": _FEET},
    )
    R.capture_point_touchdown.func = mdp.capture_point_touchdown_anchor
    R.capture_point_touchdown.params["gate_dist"] = 0.05
    R.feet_air_time_step.params["free_when_displaced_m"] = 0.05
    R.anchor_hold.params["deadband"] = 0.05
    P.anchor_point.func = mdp.anchor_point_b_held
    P.anchor_point.params.update({"hold_range_s": (0.5, 1.5), "dropout_prob": 0.05, "dropout_range_s": (2.0, 6.0)})
    cfg.actions.JointPositionAction.clip = None
    R.foot_impact_force = RewTerm(
        func=mdp.foot_contact_force_hinge, weight=-0.005,
        params={"threshold_n": 950.0, "max_excess_n": 1000.0,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )
    R.upright_bonus.params["std"] = 0.035
    R.torso_stability_bonus = None
    R.torso_stability_lin = RewTerm(func=mdp.torso_stability_bonus, weight=3.0,
                                    params={"std_lin": 0.15, "std_ang": 1000000.0, "asset_cfg": _TORSO})
    R.torso_stability_ang = RewTerm(func=mdp.torso_stability_bonus, weight=3.0,
                                    params={"std_lin": 1000000.0, "std_ang": 0.30, "asset_cfg": _TORSO})
    R.torso_lin_vel_xy.weight = -8.0
    R.foot_stance_tracking.params["std"] = 0.15
    # the heading channel (dp8_yaw -> dp8b_yaw2 -> dp8c_yaw contract, 93 obs): appended LAST, as the jobs did
    P.anchor_yaw = ObsTerm(func=mdp.anchor_yaw_b_held,
                           params={"noise_std": 0.02, "hold_range_s": (0.5, 1.5), "dropout_prob": 0.05, "dropout_range_s": (2.0, 6.0)})
    cfg.observations.critic.anchor_yaw = ObsTerm(func=mdp.anchor_yaw_b, params={"noise_std": 0.0})

    # ── 2. WORK only: no reset distances, frozen anchor ──
    cfg.events.seed_offset = None
    cfg.events.move_anchor = None

    # ── 3. the BALANCE carry list (HipsArm = Stance = Kitchen + the p14 laws) ──
    R.action_cap = RewTerm(func=mdp.action_magnitude_over, weight=-0.05,
                           params={"threshold": 1.5, "square": True, "max_excess": 5.0})
    R.knee_torque_cap = RewTerm(func=mdp.joint_torque_over_limit, weight=-0.2,
                                params={"limit_nm": 100.0, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])})
    R.feet_too_far = RewTerm(func=mdp.feet_too_far, weight=-8.0, params={"threshold": 0.45, "asset_cfg": _FEET})
    R.feet_crossed = RewTerm(func=mdp.feet_crossed, weight=-8.0, params={"min_gap": 0.05})
    R.foot_displacement_l2_from_spawn = RewTerm(func=mdp.foot_displacement_l2_from_spawn, weight=-0.375,
                                                params={"asset_cfg": _FEET})
    R.joint_vel_reg = RewTerm(func=mdp.joint_vel_l2, weight=-0.01, params={})
    R.feet_too_near.params["threshold"] = 0.25
    R.base_height = RewTerm(func=mdp.base_height_lean_l2, weight=DESK8_BASE_HEIGHT_W,
                            params={"target_height": DESK8_BASE_HEIGHT_TARGET, "command_name": "lean_command"})

    # ── 4. the BALANCE push regime ──
    cfg.events.push_robot.interval_range_s = (8.0, 12.0)
    cfg.events.sustained_push_apply.interval_range_s = (15.0, 25.0)
    cfg.curriculum.push_velocity.params["levels"] = DESK8_PUSH_LEVELS
    cfg.curriculum.sustained_push.params["hold_steps_per_level"] = (3000, 4000, 4000, 4000)
    cfg.curriculum.sustained_push.params["levels"] = DESK8_SUSTAINED_LEVELS

    # ── 5. the HipsArm arm envelope ──
    C = cfg.commands.arm_pose_command
    C.resampling_time_range = DESK8_ARM_RESAMPLING
    C.max_joint_speed = DESK8_ARM_MAX_JOINT_SPEED
    C.orientation_delta_rpy = DESK8_ARM_ORIENTATION_DELTA_RPY
    C.orientation_prob = DESK8_ARM_ORIENTATION_PROB

    # ── 6. the real-hand body ──
    _use_real_hand(cfg)

    # ── asserts: the trunk is what this docstring says ──
    assert cfg.events.seed_offset is None and cfg.events.move_anchor is None, "[Desk8-Work] transit off, anchor frozen"
    assert R.torso_stability_bonus is None and R.torso_stability_lin.weight == 3.0 and R.torso_stability_ang.weight == 3.0, "[Desk8-Work] torso split"
    assert R.base_height.weight == -15.0 and R.base_height.func is mdp.base_height_lean_l2, "[Desk8-Work] lean-aware base_height at -15"
    assert R.action_cap.params.get("square") is True and R.action_cap.weight == -0.05, "[Desk8-Work] squared action price"
    assert R.feet_too_near.params["threshold"] == 0.25 and R.feet_too_far.weight == -8.0 and R.feet_crossed.weight == -8.0, "[Desk8-Work] stance walls"
    assert R.foot_displacement_l2_from_spawn.weight == -0.375 and R.joint_vel_reg.weight == -0.01 and R.knee_torque_cap.weight == -0.2, "[Desk8-Work] carry prices"
    assert getattr(R, "energy", None) is None, "[Desk8-Work] energy is an axis row, not trunk"
    assert R.anchor_hold.params["deadband"] == 0.05 and R.torso_lin_vel_xy.weight == -8.0 and R.foot_stance_tracking.params["std"] == 0.15, "[Desk8-Work] dp8c keepers"
    assert R.upright_bonus.weight == 8.0 and R.upright_bonus.params["std"] == 0.035, "[Desk8-Work] tightened lean tracker"
    assert R.joint_deviation_hips.func is mdp.joint_deviation_l1_lean_gated and R.joint_deviation_hips.params["lean_ref"] == 0.35, "[Desk8-Work] the hinge"
    assert P.anchor_yaw.func is mdp.anchor_yaw_b_held and cfg.observations.critic.anchor_yaw.func is mdp.anchor_yaw_b, "[Desk8-Work] heading channel"
    assert tuple(cfg.curriculum.push_velocity.params["levels"])[-1] == 1.5 and tuple(cfg.events.push_robot.interval_range_s) == (8.0, 12.0), "[Desk8-Work] BALANCE push regime"
    assert tuple(C.resampling_time_range) == (1.0, 5.0) and C.max_joint_speed == 1.2 and tuple(C.orientation_delta_rpy) == (3.14, 1.5, 1.5) and C.orientation_prob == 0.9, "[Desk8-Work] HipsArm arm envelope"
    assert C.orientation_mode is True, "[Desk8-Work] orientation mode"


def _desk8_stack(cfg):
    _make_desk(cfg)
    _make_desk5(cfg)
    _make_desk5b(cfg)
    _make_desk6(cfg)
    _make_desk6b(cfg)
    _make_desk7(cfg)
    _make_desk8_work(cfg)


@configclass
class RobotEnvCfgDesk8Work(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _desk8_stack(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgDesk8Work(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _desk8_stack(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
