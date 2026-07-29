"""Unitree-H1_2-Grasp — Inspire FTP tactile grasp (gr line, 2026-07-28).

Scene: the LEFT Inspire FTP hand alone (extracted from
Aspired_Robot_Project's h1_2_with_FTP_hand.urdf: 17 tactile pad links,
6 real motors + 6 coupled followers), fixed base, PALM DOWN; the real cube
(90x60x55 mm box, ~185 g) on a kinematic support platform below it.

Episode: policy closes the fingers (6 actions, real SDK motor order) ->
at t ~ U[3,5] s the platform retracts -> the grasp alone must hold the cube
to episode end (12 s). 30 s endurance is the EVAL bar (play mode), not the
training episode.

NOTE on the robot model: this scene contains ONLY the hand — the H1-2
body never spawns, so the SYM/comx06 question does not arise here. The hand
asset is resolved via ROBOT_ASSETS_DIR/robot/inspire_ftp_left_hand/ (works
on the local box and the H200 alike).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as base_mdp

from . import grasp_mdp


@configclass
class HandArticulationCfg(ArticulationCfg):
    """ArticulationCfg + joint_sdk_names (export_deploy_cfg contract).

    For the hand line the SDK order is the REAL Inspire motor order
    (angle_set[0..5]); the followers are mechanically coupled and have no
    motor, so they are deliberately absent (export_deploy_cfg excludes
    joints not in the SDK list)."""

    joint_sdk_names: list = None

HAND_URDF = os.path.join(
    os.environ.get("ROBOT_ASSETS_DIR", os.path.expanduser("~/Projects/robot_projects/assets")),
    "robot/inspire_ftp_left_hand/inspire_ftp_left_hand.urdf",
)

# Real cube (Aspired Collision_publisher.py): 90 x 60 x 55 mm, ~170-200 g.
CUBE_SIZE = (0.090, 0.060, 0.055)
CUBE_MASS_NOMINAL = 0.185
CUBE_MASS_RANGE = (0.12, 0.25)  # operator spec 2026-07-28: ~200 g / 170 g, DR +-50 g

HAND_POS = (0.0, 0.0, 0.5)
# palm normal is link +y, fingers along link +z -> Rx(-90deg): palm DOWN, fingers +y
HAND_ROT = (0.7071068, -0.7071068, 0.0, 0.0)


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )

    robot = HandArticulationCfg(
        joint_sdk_names=grasp_mdp.DRIVER_JOINTS,
        prim_path="{ENV_REGEX_NS}/Hand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=HAND_URDF,
            fix_base=True,
            merge_fixed_joints=False,  # the 17 pads must stay separate bodies
            convert_mimic_joints_to_normal_joints=True,  # importer drops <mimic>; coupling lives in the action term
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=10.0, damping=0.2
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=HAND_POS, rot=HAND_ROT, joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
        ),
        actuators={
            # G1_INSPIRE_FTP_CFG values (flexibility-focused for grasping)
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=30.0,
                velocity_limit_sim=10.0,
                stiffness=10.0,
                damping=0.2,
                armature=0.001,
            ),
        },
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=CUBE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS_NOMINAL),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.7
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.3, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.12, 0.42)),
    )

    # kinematic support the cube rests on pre-grasp; teleported away at retract
    platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        spawn=sim_utils.CuboidCfg(
            size=(0.40, 0.40, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.45)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.12, 0.40)),
    )

    pad_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Hand/.*_force_sensor.*", update_period=0.0
    )


@configclass
class ActionsCfg:
    fingers = grasp_mdp.CoupledFingerActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """BLIND actor: tactile + proprioception only (deploy has no cube pose
        once the fingers occlude it). 17 + 6 + 6 = 29."""

        pad_forces = ObsTerm(
            func=grasp_mdp.pad_forces_log, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        joint_pos = ObsTerm(
            func=grasp_mdp.driver_joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """SIGHTED critic: actor obs (clean) + privileged cube state."""

        pad_forces = ObsTerm(func=grasp_mdp.pad_forces_log)
        joint_pos = ObsTerm(func=grasp_mdp.driver_joint_pos)
        last_action = ObsTerm(func=base_mdp.last_action)
        cube_pos = ObsTerm(func=grasp_mdp.cube_pos_in_palm)
        cube_vel = ObsTerm(func=grasp_mdp.cube_vel_in_palm)
        support = ObsTerm(func=grasp_mdp.support_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    # cube physics DR
    cube_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "mass_distribution_params": CUBE_MASS_RANGE,
            "operation": "abs",
            "recompute_inertia": True,
        },
    )
    cube_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "static_friction_range": (0.4, 1.1),
            "dynamic_friction_range": (0.35, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    # scene reset: platform height (= palm-to-cube gap), cube pose, retract time
    reset_scene = EventTerm(
        func=grasp_mdp.reset_grasp_scene,
        mode="reset",
        params={
            "palm_xy": (0.0, 0.12),
            "xy_jitter": 0.03,
            # gr2: operator measured the hand can close on the cube out to 7cm max
            # (at 7cm it is fingertip-only, not a palm grasp). 8cm was OUT OF REACH,
            # so part of the old distribution was unsolvable by construction.
            # gr2: the real fix is the CEILING 0.08 -> 0.07. Operator measured the
            # hand can only close on the cube out to 7cm (fingertip-only at 7cm), so
            # part of the old distribution was unsolvable by construction.
            # The 0.02 FLOOR is kept from gr1 on measurement: dropping it pressed the
            # cube into the palm and pre-loaded the pads, taking crush from -0.0064
            # (gr1 @ it 3) to -0.3581 (gr2 @ it 3, 56x) — crush would have become the
            # dominant term and taught "do not touch the cube", exactly backwards.
            "gap_range": (0.02, 0.07),
            "cube_height": CUBE_SIZE[2],
            "retract_time_range": (2.0, 4.0),
        },
    )
    # per-control-step check: retract the support when its time comes
    retract = EventTerm(
        func=grasp_mdp.retract_support,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={},
    )


@configclass
class RewardsCfg:
    # THE dish: cube stays at the palm once the support is gone (bounded kernel)
    # gr2: 8.0 -> 40.0 (operator: a grasping policy should value grasping REALLY
    # highly). NOTE the companion change in hold_cube_bonus: the pre-retract gate
    # dropped 0.25 -> 0.05 and is now CONTACT-GATED. Raising the weight alone would
    # have made the do-nothing exploit stronger, not weaker — the platform holds the
    # cube near the palm for free, so proximity alone must not pay much.
    hold_cube = RewTerm(func=grasp_mdp.hold_cube_bonus, weight=40.0, params={"sigma": 0.06})
    # dense shaping toward force closure (thumb + opposing finger pads loaded)
    pad_arrangement = RewTerm(
        func=grasp_mdp.pad_arrangement_bonus, weight=1.0, params={"force_thr": 0.5}
    )
    # hinged anti-crush (the 80Nm-cap pattern: holding is free, crushing is not)
    crush = RewTerm(func=grasp_mdp.crush_penalty, weight=-0.05, params={"total_thr": 30.0})
    # smoothness
    # gr2: -0.05 -> -0.01. Closing the fingers cost action_rate NOW for a payoff
    # that only arrives at retract seconds later; the policy never crossed that
    # valley and froze instead (measured pad_force_sum = 0.000 N).
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cube_dropped = DoneTerm(func=grasp_mdp.cube_dropped, params={"z_threshold": 0.08})


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: GraspSceneCfg = GraspSceneCfg(num_envs=8192, env_spacing=1.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4          # 50 Hz policy over 200 Hz physics (balance-line pattern)
        self.episode_length_s = 12.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.scene.pad_sensor.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # the 30 s endurance eval bar lives here, not in training
        self.episode_length_s = 32.0
        self.events.reset_scene.params["retract_time_range"] = (2.0, 3.0)


# --------------------------------------------------------------------------
# Queue-trainable variants (same override mechanics as Balance-Desk)
# --------------------------------------------------------------------------
from unitree_rl_lab.tasks.locomotion.robots.h1_2.balance_env_cfg_queue import (  # noqa: E402
    _apply_overrides,
    _load_overrides,
)


@configclass
class RobotEnvCfgGraspQueue(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_overrides(self, _load_overrides())  # jobs win, applied last


@configclass
class RobotPlayEnvCfgGraspQueue(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_overrides(self, _load_overrides())
