"""gr5b — ARM-IN-THE-LOOP grasping (operator design 2026-08-06).

gr5 measured the placement problem (realoffset time_out 0.79 vs 0.49); gr5b
gives the policy the ACTUATOR to solve it: the hand rides a real H1-2 arm
chain and the policy owns arm joints alongside the fingers. Two variants:

  Wrist3: hand on the DISTAL chain (wrist_roll/pitch/yaw, base = elbow link).
          Root pose (0.011, -0.2695, 0.5329) quat (0.5,0.5,0.5,0.5) puts the
          hand at zero wrist joints EXACTLY in the gr5 pose -> all gr5
          geometry (palm_xy, gap, offsets) carries over unchanged.
  Arm7:   full 7-DoF arm from the torso (elbow default 1.2 rad, root solved
          so the palm again lands in the gr5 pose).

Deploy analog: ERNEST owns the journey (parks the hand), the policy owns the
last centimeters — here expressed as small joint-target authority on the arm.
Placement error arrives as ARM PARKING NOISE (reset_arm_park_error), which the
policy can now CORRECT — the unsolvable tail of gr5_aligndr/offsetdr becomes
solvable. Thumb-table clearance (the real reason for the production -24 deg
pitch, colleague 2026-08-06) is trained, not scripted: the support platform is
WIDENED to a table slab and hand-structure contact with it is penalized.
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.envs.mdp as base_mdp

from . import grasp_mdp
from unitree_rl_lab.tasks.locomotion.robots.h1_2.balance_env_cfg_queue import (
    _apply_overrides,
    _load_overrides,
)

from .grasp_env_cfg import RobotEnvCfg

_ASSETS = os.environ.get("ROBOT_ASSETS_DIR", os.path.expanduser("~/Projects/robot_projects/assets"))

WRIST3_JOINTS = ["left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
ARM7_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
ARM7_JOINTS_R = [j.replace("left_", "right_") for j in ARM7_JOINTS]


def _make_arm_variant(cfg, urdf_name, root_pos, root_rot, arm_joints, arm_defaults, action_scale, torso_guard=False, side="left"):
    # --- asset swap: hand -> hand-on-arm chain, fixed base ---
    cfg.scene.robot.spawn.asset_path = os.path.join(_ASSETS, f"robot/{urdf_name}/{urdf_name}.urdf")
    cfg.scene.robot.init_state.pos = root_pos
    cfg.scene.robot.init_state.rot = root_rot
    # REAL TABLE HEIGHT (operator 2026-08-07): platform lands at ~1.0 m (the
    # training/real desk height) because reset derives it from the live palm.
    # SELF-COLLISIONS ON: the arm must not pass through the torso/itself.
    cfg.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True
    )
    # torso-mount height DR +-5 cm (z; xy stays with the park-error DR)
    cfg.events.mount_height_dr = EventTerm(
        func=grasp_mdp.wobble_root, mode="reset",
        params={"pos_range": (0.005, 0.005, 0.05), "rot_range": 0.0},
    )
    # no overlapping patterns (Isaac forbids '.*' + explicit); exact arm names
    jp = {f"{side}_(index|middle|ring|little|thumb).*": 0.0}
    jp.update({j: arm_defaults.get(j, 0.0) for j in arm_joints})
    cfg.scene.robot.init_state.joint_pos = jp
    # arm actuators (deploy arm gains 40/3; fingers keep their groups)
    cfg.scene.robot.actuators["arm"] = ImplicitActuatorCfg(
        joint_names_expr=[f"{side}_(shoulder|elbow|wrist).*"],
        effort_limit_sim=60.0, velocity_limit_sim=6.0,
        stiffness=40.0, damping=3.0, armature=0.01,
    )
    # --- actions: small joint-target authority around the parked default ---
    cfg.actions.arm = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=arm_joints,
        scale=action_scale, use_default_offset=True,
        clip={".*": (-1.0, 1.0)},
    )
    # --- obs: the policy must see its arm (the "resolved values") ---
    cfg.observations.policy.arm_joint_pos = ObsTerm(
        func=base_mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints)},
    )
    cfg.observations.critic.arm_joint_pos = ObsTerm(
        func=base_mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints)},
    )
    # --- placement error now lives in the ARM (correctable), not the cube ---
    cfg.events.arm_park_error = EventTerm(
        func=grasp_mdp.reset_arm_park_error, mode="reset",
        params={"joint_names": arm_joints, "noise_rad": 0.08},
    )
    cfg.events.reset_scene.params["xy_placement_error"] = 0.01
    # --- the TABLE: widen the support to a slab so thumb-clearance is real ---
    cfg.scene.platform.spawn.size = (0.45, 0.45, 0.02)
    # contact filtering requires the SINGLE-body side to host the sensor:
    # sensor on the platform, filtered against the hand STRUCTURE links
    # (pads excluded on the non-thumb fingers — they touch the cube by design).
    cfg.scene.platform.spawn.activate_contact_sensors = True
    # FILTER RULE (2026-08-09): the sensor must be ONE body per env and EACH
    # filter entry must resolve to exactly ONE prim per env — a regex that
    # expands to many fails SILENTLY (physx logs "did not match the correct
    # number of entries", force_matrix_w comes back empty, the penalty pays 0
    # forever). Enumerate every link explicitly.
    # gr6e FIX (operator HUD catch 2026-08-21: "the table_hit penalty doesn't
    # hit, even though the finger is clearly dragging against the table"):
    # the filter enumerated ONLY base + the four THUMB links — the gr5-era
    # thumb-strike concern — so index/middle/ring/little drags were invisible
    # to table_hit for the entire gr5/gr6 history. All finger phalanges added.
    cfg.scene.hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Hand/{side}_base_link"]
        + [f"{{ENV_REGEX_NS}}/Hand/{side}_thumb_{i}" for i in (1, 2, 3, 4)]
        + [f"{{ENV_REGEX_NS}}/Hand/{side}_{f}_{i}"
           for f in ("index", "middle", "ring", "little") for i in (1, 2)],
        update_period=0.0,
    )
    # --- rewards: gr5 ledger + table-hit penalty + arm smoothness (bounded) ---
    cfg.rewards.table_hit = RewTerm(
        func=grasp_mdp.table_contact_penalty, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("hand_contact"), "force_thr": 1.0, "max_val": 10.0},
    )
    cfg.rewards.arm_smooth = RewTerm(
        func=grasp_mdp.joint_vel_reversal, weight=-0.25,
        params={"joint_names": arm_joints, "max_sq": 4.0},
    )
    if torso_guard:
        # arm/hand must not strike the torso (physics blocks it now; the
        # penalty teaches avoidance instead of grinding against it)
        cfg.scene.torso_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Hand/torso_link",
            filter_prim_paths_expr=[
                f"{{ENV_REGEX_NS}}/Hand/{side}_{n}"
                for n in ("shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link",
                          "elbow_link", "wrist_roll_link", "wrist_pitch_link",
                          "wrist_yaw_link", "base_link")
            ],
            update_period=0.0,
        )
        cfg.rewards.torso_hit = RewTerm(
            func=grasp_mdp.table_contact_penalty, weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("torso_contact"), "force_thr": 1.0, "max_val": 10.0},
        )
    return cfg


@configclass
class RobotEnvCfgWrist3(RobotEnvCfg):
    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        _make_arm_variant(
            self, "inspire_hand_wrist3",
            # FOREARM ATTITUDE FIX (operator 2026-08-10: "the elbow is too
            # rotated towards the table... take inspiration from gr5b_arm, it
            # already has the correct posture"). The old root only solved for a
            # palm-down HAND; the forearm came in nearly horizontal (z-comp
            # -0.12) while arm7's operator-approved posture descends at ~40 deg
            # (z -0.65) — a measured 41.2 deg mismatch. Fixed by making wrist3
            # an EXACT replica of arm7's default sub-chain: root = arm7's
            # elbow-link world pose, wrist defaults = arm7's wrist defaults.
            # Verified: palm position delta 0.0 m, palm normal identical
            # ([-0.047, 0.015, -0.999] = down), forearm identical by
            # construction.
            root_pos=(-0.09641, -0.16734, 1.26191),
            root_rot=(0.77068, -0.19917, 0.31126, 0.51914),
            arm_joints=WRIST3_JOINTS, action_scale=0.2,
            arm_defaults={"left_wrist_roll_joint": 1.3104,
                          "left_wrist_pitch_joint": 0.3114,
                          "left_wrist_yaw_joint": 0.8009},
        )
        # jobs win, applied last — but ONLY when this class IS the leaf task
        # cfg. Subclasses (Arm7Table/TaskSpace) add trunk terms AFTER this
        # post_init runs; applying overrides here would run them BEFORE the
        # trunk exists (caught 2026-08-18: gr6b set_param on retry_seed hit
        # 'not found' — and the latent half is worse: trunk terms would
        # silently OVERWRITE job deltas, violating jobs-win, for every
        # subclassed cfg). Subclasses apply overrides at their own end.
        if type(self).__name__ in ("RobotEnvCfgArm7", "RobotEnvCfgWrist3"):
            _apply_overrides(self, _load_overrides())


@configclass
class RobotEnvCfgArm7(RobotEnvCfg):
    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        _make_arm_variant(
            self, "inspire_hand_arm7",
            # PALM-FLIP FIX (operator video review 2026-08-09: "the palm is
            # facing UPWARD instead of downward"). Cause: the IK orientation
            # error used the skew-symmetric vector, which is IDENTICALLY ZERO
            # at a 180-deg rotation error — a flipped-palm solution reported
            # "converged". Re-solved with a log-map error (180-deg branch
            # handled) + a posture prior taken from the REAL robot's MoveIt
            # seed, and VERIFIED numerically: palm normal (0,+1,0)_link maps to
            # [-0.05, 0.02, -0.999] = DOWN, fingers to [-0.01, 1.0, 0.02] = +y
            # forward — matching the gr5/wrist3 reference exactly.
            root_pos=(0.10, -0.30, 1.13), root_rot=(0.7071068, 0.0, 0.0, 0.7071068),
            arm_joints=ARM7_JOINTS, arm_defaults={
                "left_shoulder_pitch_joint": -0.3782, "left_shoulder_roll_joint": -0.1090,
                "left_shoulder_yaw_joint": -0.2756, "left_elbow_joint": 1.1317,
                "left_wrist_roll_joint": 1.3104, "left_wrist_pitch_joint": 0.3114,
                "left_wrist_yaw_joint": 0.8009,
            }, action_scale=0.15, torso_guard=True,
        )
        # jobs win, applied last — but ONLY when this class IS the leaf task
        # cfg. Subclasses (Arm7Table/TaskSpace) add trunk terms AFTER this
        # post_init runs; applying overrides here would run them BEFORE the
        # trunk exists (caught 2026-08-18: gr6b set_param on retry_seed hit
        # 'not found' — and the latent half is worse: trunk terms would
        # silently OVERWRITE job deltas, violating jobs-win, for every
        # subclassed cfg). Subclasses apply overrides at their own end.
        if type(self).__name__ in ("RobotEnvCfgArm7", "RobotEnvCfgWrist3"):
            _apply_overrides(self, _load_overrides())


# ===========================================================================
# gr6 (operator redesign 2026-08-13): the PERMANENT TABLE.
# All three observed drop modes (park: table vanished pre-pickup; sigfix:
# dropped pre-retract; quiet: cube mushed around the tabletop) were artifacts
# of the retract race — the policy had to beat a random timer or fail by
# design. Reality has no vanishing tables. gr6: the support never retracts;
# THE DISH is holding the cube at the HAND'S SPAWN POINT (above the table), so
# lifting is the only way to earn and tabletop-dragging pays nothing.
# park_keep graduates into the trunk (gr5d verdict: position discipline (park)
# beat velocity discipline (quiet) — "even better at parking the cube").
# cube_dropped termination KEPT AS-IS (operator: wait on redefining it — the
# z<0.08 floor threshold can only fire if the cube leaves the table anyway).
# ===========================================================================
def _make_gr6_table(cfg):
    # the full gr5b/gr5d obs contract BAKED IN (trunk-is-task — the gr5c
    # dropped-delta accident came from these living in job files):
    cfg.rewards.pad_arrangement.params["closure_mode"] = "count"
    cfg.observations.policy.cube_pose = ObsTerm(
        func=grasp_mdp.cube_pose_vision,
        params={"noise_std_pos": 0.005, "noise_std_axis": 0.02},
        clip=(-1.0, 1.0),
    )
    # the table never leaves
    cfg.events.retract = None
    # hold target = palm spawn point, captured per-env at reset
    from isaaclab.managers import EventTermCfg as _ET
    cfg.events.hand_start = _ET(func=grasp_mdp.capture_hand_start, mode="reset")
    cfg.rewards.cube_hold_above = RewTerm(
        func=grasp_mdp.cube_hold_above_bonus, weight=25.0,
        params={"height": 0.15, "sigma": 0.06},
    )
    # gr6b no-op-by-default events; jobs enable via set_param:
    #   retry_seed.prob 0.2       (retry seeding — dp5 lean-seeding lesson)
    #   mount_orbit.amp_range ... (transport DR, randomized phase)
    cfg.events.retry_seed = _ET(
        func=grasp_mdp.reset_cube_retry, mode="reset",
        params={"prob": 0.0, "dist_range": (0.10, 0.20)},
    )
    cfg.events.mount_orbit = _ET(
        func=grasp_mdp.mount_orbit, mode="interval", interval_range_s=(0.02, 0.02),
        params={"amp_range": (0.0, 0.0), "freq_range": (0.1, 0.4)},
    )
    # gr5d verdict: park discipline in the trunk
    cfg.rewards.park_keep = RewTerm(
        func=grasp_mdp.park_keep_bonus, weight=2.0, params={"sigma": 1.7},
    )


@configclass
class RobotEnvCfgArm7Table(RobotEnvCfgArm7):
    """gr6_table: the task change ONLY (permanent table + hold-at-start),
    joint-space arm as gr5d — isolates what the new task does."""
    def __post_init__(self):
        super().__post_init__()
        _make_gr6_table(self)
        _apply_overrides(self, _load_overrides())  # jobs win, applied last


@configclass
class RobotEnvCfgArm7TaskSpace(RobotEnvCfgArm7):
    """gr6_taskspace: gr6_table + the TASK-SPACE hand interface (colleague
    convergence design, 2026-08-13): the policy commands d(x,y,z) AND
    d(orientation) of the hand — 6 relative-pose actions through Isaac Lab's
    stock DLS IK action — plus the 6 fingers. The joint-count debate (3 vs 7)
    dissolves: the policy speaks task space, the resolver owns kinematics,
    and deploy-side the same deltas go to the real IK resolver (ERNEST).
    Orientation freedom included per operator ("otherwise the thumb will hit
    the table" — the -24deg wrist-pitch lesson, now learnable in task space).
    """
    def __post_init__(self):
        super().__post_init__()
        _make_gr6_table(self)
        from isaaclab.envs.mdp.actions.actions_cfg import (
            DifferentialInverseKinematicsActionCfg as _IKAct,
        )
        from isaaclab.controllers import DifferentialIKControllerCfg as _IKCtl
        self.actions.arm = _IKAct(
            asset_name="robot",
            joint_names=ARM7_JOINTS,
            body_name="left_base_link",          # the hand base = end-effector
            controller=_IKCtl(command_type="pose", use_relative_mode=True,
                              ik_method="dls"),
            # action in [-1,1]^6 -> per-step delta: 3 cm translation,
            # 0.05 rad rotation. Small steps = the "last five centimeters"
            # contract; workspace legality is the resolver's job (rule 4/21:
            # an unreachable wish shows up as IK residual, not a crash).
            scale=(0.03, 0.03, 0.03, 0.05, 0.05, 0.05),
        )
        _apply_overrides(self, _load_overrides())  # jobs win, applied last


# ===========================================================================
# RIGHT-HAND ERA (operator ruling 2026-08-24: the real robot's LEFT hand is
# not fully working — grasp training moves to the RIGHT side).
# Asset: inspire_hand_arm7_right (single-source extraction from the vendor
# FTP full-robot URDF — factory-correct right-arm limits, 17 pads, hand mass
# fixed to the 790 g datasheet; scripts/tools/build_inspire_arm7_right.py in
# aspired-isaac-lab). grasp_mdp names are side-agnostic regexes; only the
# geometry below is handed.
# Mirror across the XZ plane: root y flips, Rz(+90deg) -> Rz(-90deg); pitch
# joints (shoulder_pitch/elbow/wrist_pitch) keep their default signs,
# roll/yaw joints flip. The table/cube move to the -y side via palm_xy.
# ===========================================================================
@configclass
class RobotEnvCfgArm7R(RobotEnvCfg):
    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        _make_arm_variant(
            self, "inspire_hand_arm7_right",
            root_pos=(0.10, 0.30, 1.13), root_rot=(0.7071068, 0.0, 0.0, -0.7071068),
            arm_joints=ARM7_JOINTS_R, arm_defaults={
                "right_shoulder_pitch_joint": -0.3782, "right_shoulder_roll_joint": 0.1090,
                "right_shoulder_yaw_joint": 0.2756, "right_elbow_joint": 1.1317,
                "right_wrist_roll_joint": -1.3104, "right_wrist_pitch_joint": 0.3114,
                "right_wrist_yaw_joint": -0.8009,
            }, action_scale=0.15, torso_guard=True, side="right",
        )
        if type(self).__name__ == "RobotEnvCfgArm7R":
            _apply_overrides(self, _load_overrides())


@configclass
class RobotEnvCfgArm7TableR(RobotEnvCfgArm7R):
    """gr7-era task: the gr6 permanent table, RIGHT hand. Table/cube on the
    -y side (mirrored palm_xy) so the right palm spawns over them exactly as
    the left palm did on +y."""
    def __post_init__(self):
        super().__post_init__()
        _make_gr6_table(self)
        self.events.reset_scene.params["palm_xy"] = (0.0, -0.12)
        _apply_overrides(self, _load_overrides())  # jobs win, applied last


# ---------------------------------------------------------------------------
# gr7c (operator 2026-08-27): gr7b_clean_smooth PROMOTED TO TRUNK ("stable and
# robust enough — from that trunk retrofit it"). Bakes the full clean_smooth
# economy on the RIGHT-hand table task: the gr6f_combo base (height-only
# cube_hold_above s0.03 @40, binary table wall -10/max2, approach_cube +4) +
# quiet_hold +5 + cube_slide -2 + arm_smooth -1. Trunk-is-task; jobs win last.
# ---------------------------------------------------------------------------
def _make_gr7b_clean_smooth(cfg):
    from unitree_rl_lab.tasks.manipulation import grasp_mdp as _gm
    cfg.rewards.cube_hold_above.weight = 40.0
    cfg.rewards.cube_hold_above.params["height_only"] = True
    cfg.rewards.cube_hold_above.params["sigma"] = 0.03
    cfg.rewards.table_hit.weight = -10.0
    cfg.rewards.table_hit.params["max_val"] = 2.0
    cfg.rewards.approach_cube = RewTerm(func=_gm.approach_cube_bonus, weight=4.0, params={"sigma": 0.3})
    cfg.rewards.quiet_hold = RewTerm(func=_gm.quiet_hold_bonus, weight=5.0, params={"sigma": 3.0, "force_thr": 0.5})
    cfg.rewards.cube_slide = RewTerm(func=_gm.cube_slide_penalty, weight=-2.0,
                                     params={"ramp_lo": 0.01, "ramp_hi": 0.05, "max_speed": 1.0})
    cfg.rewards.arm_smooth.weight = -1.0


@configclass
class RobotEnvCfgArm7TableRCS(RobotEnvCfgArm7R):
    """GraspR-Arm7Table-CS: the clean_smooth trunk (right hand)."""
    def __post_init__(self):
        super().__post_init__()
        _make_gr6_table(self)
        self.events.reset_scene.params["palm_xy"] = (0.0, -0.12)
        _make_gr7b_clean_smooth(self)
        _apply_overrides(self, _load_overrides())  # jobs win, applied last


# ---------------------------------------------------------------------------
# gr8 (operator 2026-08-27): real desk objects — the TUBE replaces the cube.
# assets/objects/tube_d180_h130: ⌀180 x 130 mm, 5 mm walls, 363 g, generator-
# built (visual revolve mesh + 16 box-segment collisions so the hole survives
# Isaac's convex pipeline; collider_type convex_hull hulls each box alone).
# GEOMETRY WARNING: the hand closes on <=7 cm objects (gr2 measurement) — an
# 18 cm tube can never be cube-wrapped; the graspable feature is the 5 mm RIM.
# Iteration 1 = ASSET SWAP ONLY (spawn + height + mass DR); rim-grasp reward
# design waits until the operator has seen the scale in a preview.
# ---------------------------------------------------------------------------
def _make_gr8_tube(cfg):
    tube_urdf = os.path.join(_ASSETS, "objects/tube_d180_h130/tube_d180_h130.urdf")
    cfg.scene.cube.spawn = sim_utils.UrdfFileCfg(
        asset_path=tube_urdf,
        fix_base=False,
        joint_drive=None,          # single link, no joints — skip drive validation
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.363),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    # reset_scene places by CENTER pose + cube_height/2 (tube origin is centered)
    cfg.events.reset_scene.params["cube_height"] = 0.13
    # RIM PRESENTATION (measured from the MuJoCo kinematic twin at arm_defaults:
    # fingers point env -y, thumb on +x): displace the tube center 9 cm (=R)
    # toward the THUMB side, so the rim WALL lies on the jaw line — fingers
    # outside, thumb over the mouth — instead of the palm hovering over the
    # open mouth (operator: hand rendered inside the tube).
    cfg.events.reset_scene.params["object_xy_offset"] = (0.09, 0.0)
    # real tube 363 g; keep the cube's ±~35% DR proportions
    cfg.events.cube_mass.params["mass_distribution_params"] = (0.30, 0.43)
    # LIFT-RAMP GEOMETRY (caught 2026-08-27 pre-wave): _lift_ramp measures the
    # object CENTER above the table, and lo/hi are tuned to the CUBE's resting
    # center (0.0275+0.002). The tube's center RESTS at 0.065 — past hi=0.05,
    # i.e. a tube sitting on the table read as fully lifted (cube_slide dead,
    # hold income mis-gated). Port the SEMANTICS: shift every ramp threshold by
    # d_half = 0.065 - 0.0275 = +0.0375, and the hold height by the same.
    # TILTGATE LAW (gr6c, an order worse here: radius 0.09 > half-height
    # 0.065): a TIPPED tube's center reaches h=0.09 — income must gate ABOVE
    # anything reachable by tilting, or tipping reads as lifting. lo 0.095 >
    # 0.09; full ramp at +7 cm of true lift; hold target above full-ramp.
    for _term in ("hold_cube", "cube_hold_above", "cube_slide"):
        getattr(cfg.rewards, _term).params["ramp_lo"] = 0.095
        getattr(cfg.rewards, _term).params["ramp_hi"] = 0.135
    cfg.rewards.cube_hold_above.params["height"] = 0.20


@configclass
class RobotEnvCfgArm7TableRCSTube(RobotEnvCfgArm7TableRCS):
    """GraspR-Arm7Table-CS-Tube: the CS trunk with the real tube as the object."""
    def __post_init__(self):
        super().__post_init__()
        _make_gr8_tube(self)
        _apply_overrides(self, _load_overrides())  # jobs win, applied last


def _make_gr8_ring(cfg):
    """The ⌀181-191 x 20 mm collar ring (52 g). Trained SEPARATELY from the
    tube (operator 2026-08-27: no generalizing over both — different weights).
    Grasp feature = pinching the 5 mm wall / 10 mm roof lip."""
    ring_urdf = os.path.join(_ASSETS, "objects/ring_d180_h20/ring_d180_h20.urdf")
    cfg.scene.cube.spawn = sim_utils.UrdfFileCfg(
        asset_path=ring_urdf,
        fix_base=False,
        joint_drive=None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.052),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    cfg.events.reset_scene.params["cube_height"] = 0.02
    cfg.events.cube_mass.params["mass_distribution_params"] = (0.042, 0.062)
    # rim presentation: same wrist-frame jaw-line displacement as the tube
    cfg.events.reset_scene.params["object_offset_wrist_frame"] = (0.0, 0.0, 0.10)
    cfg.events.reset_scene.params["palm_track"] = True
    # ring is only 2 cm tall: gap floor = the fingertip drop (~0.065) so the
    # tips spawn AT ring-top height, never through the ring into the platform
    cfg.events.reset_scene.params["gap_range"] = (0.065, 0.09)
    # TILTGATE LAW (see _make_gr8_tube): a ring stood on its RIM has center
    # h~0.0905 — lo must clear it or edge-standing farms the hold income
    # (rest center is only 0.012, so the ring pays nothing until a real
    # 10 cm lift; the honest price of a tall-when-tipped object).
    for _term in ("hold_cube", "cube_hold_above", "cube_slide"):
        getattr(cfg.rewards, _term).params["ramp_lo"] = 0.10
        getattr(cfg.rewards, _term).params["ramp_hi"] = 0.14
    cfg.rewards.cube_hold_above.params["height"] = 0.16


@configclass
class RobotEnvCfgArm7TableRCSRing(RobotEnvCfgArm7TableRCS):
    """GraspR-Arm7Table-CS-Ring: the CS trunk with the real ring as the object."""
    def __post_init__(self):
        super().__post_init__()
        _make_gr8_ring(self)
        _apply_overrides(self, _load_overrides())  # jobs win, applied last
