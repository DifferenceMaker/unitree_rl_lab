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
    cfg.scene.hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Hand/left_base_link",
                                "{ENV_REGEX_NS}/Hand/left_thumb_.*"],
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
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Hand/left_(shoulder|elbow|wrist).*",
                                    "{ENV_REGEX_NS}/Hand/left_base_link"],
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
            root_pos=(0.011, -0.2695, 1.1129), root_rot=(0.5, 0.5, 0.5, 0.5),
            arm_joints=WRIST3_JOINTS, arm_defaults={}, action_scale=0.2,
        )
        _apply_overrides(self, _load_overrides())  # jobs win, applied last


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
        _apply_overrides(self, _load_overrides())  # jobs win, applied last
