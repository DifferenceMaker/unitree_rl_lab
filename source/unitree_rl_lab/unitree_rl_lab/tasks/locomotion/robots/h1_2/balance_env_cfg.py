import math

import isaaclab.sim as sim_utils
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_H1_2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp


LEGS_TORSO_JOINT_REGEX = [
    ".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint",
    "torso_joint",
]
ARM_JOINT_REGEX = [
    ".*_shoulder_pitch.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*",
    ".*_elbow.*", ".*_wrist.*",
]


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat-ground scene for H1-2 option (ii) Phase 2 training."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Domain randomization and reset events.
    Arms commanded externally by UniformArmPoseCommand with apply_directly=True
    (Phase 2 extension of Phase 1 mechanism — no event term needed)."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.0),
            "dynamic_friction_range": (0.7, 1.0),
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

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Phase A scope: pushes disabled
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )


@configclass
class CurriculumCfg:
    """Phase 2 curriculum: pitch fixed, roll amplitude ramps."""

    arm_pose = CurrTerm(
        func=mdp.arm_pose_curriculum_phase2,
        params={
            "command_term_name": "arm_pose_command",
            "warmup_steps": 6000,
            "hold_steps": 24000,                              # ~1000 iters per level
            "pitch_amplitude": 1.5,                           # stays at Phase 1 max
            "roll_amplitude_levels": (0.0, 0.3, 0.6, 1.0),
        },
    )


@configclass
class CommandsCfg:
    """Standing velocity (zero) + bilateral pitch + mirrored roll disturbance."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
    )

    arm_pose_command = mdp.UniformArmPoseCommandCfg(
        asset_name="robot",
        all_arm_joint_names=ARM_JOINT_REGEX,
        pitch_amplitude=1.5,                # set by curriculum
        roll_amplitude=0.0,                 # set by curriculum
        apply_directly=True,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Policy commands ONLY legs+torso (13 joints).
    Arm joints get targets from arm_pose_command."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEGS_TORSO_JOINT_REGEX,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        # Policy observes commanded arm pose (14-dim) so it can anticipate
        # both pitch and roll disturbances.
        arm_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "arm_pose_command"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01)
        last_action = ObsTerm(func=mdp.last_action)
        arm_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "arm_pose_command"}
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Same rewards as Phase 1 — balance only, no arm tracking."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    alive = RewTerm(func=mdp.is_alive, weight=20.0)

    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEGS_TORSO_JOINT_REGEX)},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEGS_TORSO_JOINT_REGEX)},
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["torso_joint"])},
    )

    joint_deviation_hips = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    stance_bonus_legs_torso = RewTerm(
        func=mdp.stance_bonus,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint",
                "torso_joint",
            ]),
            "std": 0.15,
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 1.0})

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["torso_link", ".*hip.*", ".*knee.*", ".*shoulder.*", ".*elbow.*"])},
    )

    torso_lin_vel_xy = RewTerm(
        func=mdp.body_lin_vel_xy_l2,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    torso_ang_vel = RewTerm(
        func=mdp.body_ang_vel_l2,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link"]),
            "threshold": 50.0,
        },
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
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
        self.episode_length_s = 60.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Force max disturbance at play time (curriculum would otherwise
        # lock at level 0 because common_step_counter starts at 0).
        self.curriculum.arm_pose.params["roll_amplitude_levels"] = (1.0, 1.0, 1.0, 1.0)
        self.commands.arm_pose_command.pitch_amplitude = 1.5
        self.commands.arm_pose_command.roll_amplitude = 1.0
