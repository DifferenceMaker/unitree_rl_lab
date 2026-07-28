import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_H1_2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# H1-2 joint regexes (verified against UNITREE_H1_2_CFG / h1_2.urdf).
# 13 leg+torso + 14 arm = 27 actions. Excludes Inspire finger joints ([LR]_.* hands).
LEGS_TORSO_JOINT_REGEX = [
    ".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "torso_joint",
]
ARM_JOINT_REGEX = [
    ".*_shoulder_pitch.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*",
    ".*_elbow.*", ".*_wrist.*",
]
WALK_JOINT_REGEX = LEGS_TORSO_JOINT_REGEX + ARM_JOINT_REGEX

FOOT_LINK = ".*_ankle_roll_link"  # the actual foot body (not ankle_pitch_link)

GAIT_PERIOD = 0.75


COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),  # flat-only v1
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events. Upstream mild DR + pushes, retained."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications. 27-action full-body, fingers excluded."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=WALK_JOINT_REGEX, scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications. H1-style: gait_phase obs, no history."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": GAIT_PERIOD})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01)
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": GAIT_PERIOD})

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_REGEX)},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["torso_joint"])},
    )
    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 1.0})  # TODO verify in sim

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": GAIT_PERIOD,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=20.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK),
        },
    )
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.0002,
        params={"threshold": 500, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK)},
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle_roll.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link"]),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the H1-2 locomotion velocity-tracking environment."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.gait_f = 1.0 / GAIT_PERIOD

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


# =====================================================================
# lm1 (2026-07-28): queue-trainable walk, pipeline-aligned.
#
# The reward set and BOTH curricula (terrain_levels_vel, lin_vel_cmd_levels)
# are DELIBERATELY UNCHANGED from locomotion_v1 — this run is the reference
# point we will add to, not a redesign. See the balance-vs-walk audit: the
# 14 balance-only terms are anti-motion machinery (heading/pos-from-spawn,
# foot_stance_tracking, feet_air_time_step, capture_point_touchdown, ...)
# and are intentionally ABSENT here.
#
# The only pipeline changes:
#   1. ASSET PINNED EXPLICITLY to h1_2.urdf (the SYM body, 77.27 kg) — the
#      same file the desk line trains on. Without this, UNITREE_H1_2_CFG on
#      aspired/training resolves to h1_2_comx06.urdf and the walk task would
#      SILENTLY train on the wrong CoM belief with nothing in the logs to
#      say so. comx06 (+5.6mm) was the barefoot-era standard; SYM (+22.4mm)
#      is the hardware-validated shod CoM (2026-07-23 ankle-torque measurement).
#   2. QUEUE_JOB_JSON overrides applied LAST, same helpers as Balance-Desk,
#      so .job files can tune weights/params/raw lines without code edits.
#
# experiment_name stays "" (-> derives to unitree_h1_2_walk), so walk runs
# get their OWN log dir and never mix with the balance warmstart lineage.
# Harvest with: harvest_done.sh --task unitree_h1_2_walk
# =====================================================================
from .balance_env_cfg_queue import _apply_overrides, _load_overrides  # noqa: E402

SYM_URDF = "h1_2.urdf"
COMX06_URDF = "h1_2_comx06.urdf"

# The hardware-validated shod CoM, measured 2026-07-23 by ankle torque + lean.
# NOTE (2026-07-28): the file NAME is not enough to identify the body. There are
# two different h1_2.urdf files in play:
#   <assets>/robot/h1_2/h1_2.urdf                        torso x=0.0155 y=0.002797  (STOCK Unitree)
#   <assets>/robot/h1_2/sweep/SYM/robot/h1_2/h1_2.urdf   torso x=0.03   y=0.0       (SYM)
# Only the second is SYM. So we verify the CONTENTS, not the filename, and say
# so in the log — a silent CoM mismatch has no other symptom in training.
SYM_TORSO_X = 0.03
SYM_TORSO_Y = 0.0


def _torso_com(urdf_path):
    """(x, y) of torso_link's inertial origin, or None if unreadable."""
    import re
    try:
        src = open(urdf_path).read()
    except OSError:
        return None
    link = re.search(r'<link\s+name="torso_link".*?</link>', src, re.S)
    if not link:
        return None
    o = re.search(r'<inertial>.*?<origin[^>]*xyz="([^"]+)"', link.group(0), re.S)
    if not o:
        return None
    parts = o.group(1).split()
    return float(parts[0]), float(parts[1])


def _pin_sym_asset(cfg) -> None:
    """Resolve the SYM body and VERIFY it, loudly.

    1. swap comx06 -> h1_2.urdf (the barefoot-era default on aspired/training)
    2. if a sweep/SYM tree sits alongside, prefer it — that is the real SYM body
       and the one p12k_desk_anchor2 trained on
    3. read torso_link's inertial origin and print it; warn if it is not SYM
    """
    import os

    path = cfg.scene.robot.spawn.asset_path
    if COMX06_URDF in path:
        path = path.replace(COMX06_URDF, SYM_URDF)

    # Prefer an adjacent sweep/SYM body if one exists (H200 layout).
    h1_2_dir = os.path.dirname(path)
    sweep = os.path.join(h1_2_dir, "sweep", "SYM", "robot", "h1_2", SYM_URDF)
    if os.path.exists(sweep):
        path = sweep

    cfg.scene.robot.spawn.asset_path = path
    com = _torso_com(path)
    if com is None:
        print(f"[walk] !! robot asset: {path} (could NOT read torso CoM to verify)")
        return
    x, y = com
    ok = abs(x - SYM_TORSO_X) < 1e-4 and abs(y - SYM_TORSO_Y) < 1e-4
    tag = "SYM (hardware-validated shod CoM)" if ok else "!! NOT SYM !!"
    print(f"[walk] robot asset: {path}")
    print(f"[walk] torso CoM x={x:+.4f} y={y:+.4f} -> {tag}")
    if not ok:
        print(
            f"[walk] !! WARNING: expected SYM torso x={SYM_TORSO_X} y={SYM_TORSO_Y}. "
            f"Point ROBOT_ASSETS_DIR at the sweep/SYM tree "
            f"(<assets>/robot/h1_2/sweep/SYM) to train on the desk-line body."
        )


@configclass
class RobotEnvCfgWalkQueue(RobotEnvCfg):
    """Queue-trainable walk task (Unitree-H1_2-Walk-Q)."""

    def __post_init__(self):
        super().__post_init__()
        _pin_sym_asset(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgWalkQueue(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _pin_sym_asset(self)
        _apply_overrides(self, _load_overrides())
