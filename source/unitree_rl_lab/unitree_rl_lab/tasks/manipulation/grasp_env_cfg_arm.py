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


def _make_arm_variant(cfg, urdf_name, root_pos, root_rot, arm_joints, arm_defaults, action_scale, torso_guard=False):
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
    jp = {"left_(index|middle|ring|little|thumb).*": 0.0}
    jp.update({j: arm_defaults.get(j, 0.0) for j in arm_joints})
    cfg.scene.robot.init_state.joint_pos = jp
    # arm actuators (deploy arm gains 40/3; fingers keep their groups)
    cfg.scene.robot.actuators["arm"] = ImplicitActuatorCfg(
        joint_names_expr=["left_(shoulder|elbow|wrist).*"],
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
    cfg.scene.hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Hand/left_base_link"]
        + [f"{{ENV_REGEX_NS}}/Hand/left_thumb_{i}" for i in (1, 2, 3, 4)],
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
                f"{{ENV_REGEX_NS}}/Hand/left_{n}"
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
