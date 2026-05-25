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

# Body joints only — excludes Inspire FTP-12 hand finger joints.
# Matches Unitree HG LowState_ protocol (27 body motors). Hand state
# lives on separate HandState_ channel. See session 2026-05-18.
BODY_JOINT_REGEX = LEGS_TORSO_JOINT_REGEX + ARM_JOINT_REGEX


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat-ground scene for H1-2 Phase 5 v2 training (wobble + push merge)."""

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
    """Phase 5 v2 events: phase4's reset/material/mass events + impulse push
    (now non-zero from iter 0, ramped via push_velocity curriculum) +
    sustained external force events from p5_v1.

    push_robot.velocity_range and sustained_push_apply.params are both
    overridden each step by their respective curriculum terms — the initial
    values here are just the starting points at step 0.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),       # v3 DR: was (0.7, 1.0), now symmetric ±30% of nominal
            "dynamic_friction_range": (0.7, 1.3),      # v3 DR: was (0.7, 1.0)
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-3.0, 5.0),   # v3 DR: was (-1.0, 3.0), wider but keeps asymmetry
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

    # Discrete impulse push — initial range is the curriculum's level 0 (0.30 m/s).
    # push_velocity_curriculum overrides velocity_range each step.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )

    # Sustained external force on torso — magnitude and duration controlled
    # by sustained_push_curriculum each step (zero during warmup).
    sustained_push_apply = EventTerm(
        func=mdp.apply_sustained_external_force,
        mode="interval",
        interval_range_s=(15.0, 25.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_magnitude_range": (0.0, 0.0),    # overridden by curriculum
            "duration_range_s": (0.0, 0.0),         # overridden by curriculum
        },
    )

    # Companion to sustained_push_apply — clears expired forces each policy step.
    sustained_push_clear = EventTerm(
        func=mdp.clear_expired_sustained_pushes,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # = policy step dt
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )


@configclass
class CurriculumCfg:
    """Phase 5 v2 curriculum: wobble pinned at max (warmstart from phase4_v5),
    push_velocity ramped 0.30 → 2.0 m/s, sustained_push ramped 0N → 50N.

    Wobble curriculum is kept structurally (so the env builds with
    arm_pose_curriculum_phase4 still in the pipeline) but all amplitude
    levels = 0.15 from iter 0 — the warmstart policy already survives this.
    """

    arm_pose = CurrTerm(
        func=mdp.arm_pose_curriculum_phase4,
        params={
            "command_term_name": "arm_pose_command",
            "warmup_steps": 0,                        # no warmup — start at max
            "hold_steps": 24000,                       # unused (all levels equal)
            "pitch_amplitude": 1.5,                    # held pose pitch (constant)
            "roll_amplitude": 1.0,                     # held pose roll (constant)
            "elbow_amplitude": 1.5,                    # held pose elbow (constant)
            "wobble_amplitude_levels": (0.15, 0.15, 0.15, 0.15),
        },
    )

    push_velocity = CurrTerm(
        func=mdp.push_velocity_curriculum,
        params={
            "event_term_name": "push_robot",
            "warmup_steps": 2000,
            "hold_steps": 12000,           # ~500 iters per level
            "levels": (
                0.30, 0.36, 0.43, 0.51,   # gentler initial steps (~20% jumps)
                0.61, 0.73, 0.87, 1.04,
                1.24, 1.48, 1.76, 2.00,
            ),
        },
    )

    sustained_push = CurrTerm(
        func=mdp.sustained_push_curriculum,
        params={
            "event_term_name": "sustained_push_apply",
            "warmup_steps": 3000,         # ~125 iter — re-establish before ramping
            # Per-level dwell time. Doubles at the regime where v2c diverged
            # (40N+) to give the critic more time to converge to the new
            # return distribution. Total ramp ~3650 iters to reach 75N max.
            "hold_steps_per_level": (
                7200,    # warmup → 15N    (~300 iter, was 300)
                7200,    # 15N → 20N        (~300 iter, was 300)
                7200,    # 20N → 25N        (~300 iter, was 300)
                9600,    # 25N → 32N        (~400 iter, +33%)
                12000,   # 32N → 40N        (~500 iter, +67%)
                14400,   # 40N → 50N        (~600 iter, +100%) ← divergence zone
                16800,   # 50N → 60N        (~700 iter, +133%)
                19200,   # 60N → 75N        (~800 iter, +167%) ← max regime
            ),
            "levels": (
                ((0.0,  0.0),  (0.0, 0.0)),   # warmup
                ((5.0,  15.0), (1.0, 2.0)),
                ((8.0,  20.0), (1.2, 2.5)),
                ((10.0, 25.0), (1.5, 3.0)),
                ((12.0, 32.0), (1.7, 3.3)),
                ((15.0, 40.0), (2.0, 3.5)),
                ((18.0, 50.0), (2.0, 4.0)),
                ((22.0, 60.0), (2.5, 4.0)),
                ((28.0, 75.0), (3.0, 4.0)),
            ),
        },
    )



@configclass
class CommandsCfg:
    """Standing velocity (zero) + held arm pose with wobble."""

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
        pitch_amplitude=1.5,                # set by curriculum (constant)
        roll_amplitude=1.0,                 # set by curriculum (constant)
        elbow_amplitude=1.5,                # set by curriculum (constant)
        wobble_amplitude=0.0,               # set by curriculum (→ 0.15)
        wobble_frequency=2.0,               # 2 Hz
        apply_directly=True,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Policy commands ONLY legs+torso (13 joints). Arms driven externally
    by arm_pose_command (Option II architecture)."""

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
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_REGEX, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_REGEX, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action)
        # Observes held pose only (wobble not in obs — it's a small reactive disturbance)
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
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_REGEX, preserve_order=True)},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_REGEX, preserve_order=True)},
        )
        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_REGEX, preserve_order=True)},
        )
        last_action = ObsTerm(func=mdp.last_action)
        arm_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "arm_pose_command"}
        )

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Phase 5 v2 rewards: phase4's reward set (kept verbatim) + p5_v1's
    foot_stance_tracking. No reverts of phase4's deliberate exclusions:
    - undesired_contacts stays scoped to torso/hip/knee only
      (NO shoulder/elbow — arms are not policy-controlled, can't be blamed)
    - base_contact termination stays REMOVED (was triggering 19% false
      terminations from arm-to-torso self-contact in Phase 3)
    """

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

    # NEW (from p5_v1): exponential bonus for keeping feet near nominal stance.
    # Creates "small steps cheap, big steps expensive" gradient via exp(-d/std).
    foot_stance_tracking = RewTerm(
        func=mdp.foot_stance_tracking,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "std": 0.08,
            "nominal_foot_pos_b": [[0.0, 0.10], [0.0, -0.10]],
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 1.0})

    # NOTE: undesired_contacts kept to phase4 scope. NO shoulder/elbow:
    # arm-to-torso self-contact during wobble would penalize policy for
    # something it can't control.
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["torso_link", ".*hip.*", ".*knee.*"])},
    )

    # Kept at Phase 3 value (doubling destabilized PPO — see 2026-05-14 log)
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

    # Positive bonus for camera stability — structural fix for the
    # "compromise" trap from doubled torso penalties
    torso_stability_bonus = RewTerm(
        func=mdp.torso_stability_bonus,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "std_lin": 0.10,	# 0.15 -> 0.10 for v2_3. Tighter sway tolerance.
            "std_ang": 0.30,
        },
    )


@configclass
class TerminationsCfg:
    """Phase 5 v2 terminations: phase4's set kept verbatim.

    Per 2026-05-14 lessons:
    - base_contact was REMOVED. Threshold-raising attempts destabilized
      PPO during warmstart. Falls still caught via base_height (pelvis < 0.5m).
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})


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
        self.episode_length_s = 60.0  # robot must stand indefinitely
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # Bumped 2× for high-force regime (v2c divergence investigation).
        # At 50N+ sustained on torso, the robot visits unusual configurations
        # that may produce more contact pairs than default buffers handle.
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Play-time env: forces max disturbances on all curricula so visual
    evaluation matches the training distribution at max difficulty."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32

        # Force max wobble (already 0.15 in train, kept consistent)
        self.curriculum.arm_pose.params["wobble_amplitude_levels"] = (0.15, 0.15, 0.15, 0.15)
        self.commands.arm_pose_command.pitch_amplitude = 1.5
        self.commands.arm_pose_command.roll_amplitude = 1.0
        self.commands.arm_pose_command.elbow_amplitude = 1.5
        self.commands.arm_pose_command.wobble_amplitude = 0.15

        # Force max impulse push velocity (skip ramp)
        self.curriculum.push_velocity.params["warmup_steps"] = 0
        self.curriculum.push_velocity.params["levels"] = (2.00,) * 9
        self.events.push_robot.params["velocity_range"] = {
            "x": (-2.0, 2.0), "y": (-2.0, 2.0),
        }

        # Force max sustained push (skip warmup, top level)
        self.curriculum.sustained_push.params["warmup_steps"] = 0
        self.curriculum.sustained_push.params["levels"] = (
            ((15.0, 50.0), (2.0, 4.0)),
        ) * 4
        self.events.sustained_push_apply.params["force_magnitude_range"] = (15.0, 50.0)
        self.events.sustained_push_apply.params["duration_range_s"] = (2.0, 4.0)
