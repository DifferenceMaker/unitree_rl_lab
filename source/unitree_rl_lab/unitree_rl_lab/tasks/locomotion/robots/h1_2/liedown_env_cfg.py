"""Unitree-H1_2-LieDown v1 — safety lie-down policy.

When triggered (overheat / low battery), the robot lowers itself from its
standing balance stance to lying supine on the ground: slowly, controlled,
soft. Trained from scratch (no warmstart — the task shares no reward
economy with the balance line).

Architecture decisions (paper survey 2026-07-28, memory
liedown-policy-paper-survey):

- HoST reversed: dense height-descent task reward, orientation reward gated
  below 0.5 m, "quiet lying" post-task group gated on low+supine, their
  regularization group. NO curriculum — gravity does the exploration work
  that HoST's force curriculum existed for.
- β action governor (HoST): RelativeJointPositionAction, PD target =
  measured q + clamp(0.3·a, ±0.3 rad). Bounds the max PD-error torque
  kick to β·kp per joint and thereby the motion speed. Deploy-side note:
  the runner must implement the same delta-from-measured-q transform
  (differs from the balance line's absolute default-offset actions!).
- FIRM damage triplet: contact-force yank, body momentum rate, per-group
  impact-force excess. Minimal terminations — NO base-height, NO non-foot
  contact termination (lying IS the goal); only ballistic velocity and
  forward-faceplant orientation terminate.
- SafeFall body groups: torso (head+cameras live there — H1-2 URDF has no
  separate head link) protected hardest; arms next (40 Nm ceiling, keep
  load in the legs); knees/hips cheap; pelvis cheap above a slam threshold.
- Full-body 27-action (walk-task convention): the arms must brace/settle,
  they cannot stay on an external command channel like the balance line.

v2 backlog (deliberately NOT in v1):
- Init from balance-policy rollout states + FIRM zero-actuation dropout
  ("from any posture" trigger coverage).
- Phase-2 slow-down: harvest v1 trajectory, stretch 1.5–8x, retrain as
  tracking task (Contact-Staged Stand-up: 0/10 sim2sim without this step;
  our insurance if v1's descent is too fast or ugly).
- Small pushes during descent for robustness.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
    ".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "torso_joint",
]
ARM_JOINT_REGEX = [
    ".*_shoulder_pitch.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*",
    ".*_elbow.*", ".*_wrist.*",
]
BODY_JOINT_REGEX = LEGS_TORSO_JOINT_REGEX + ARM_JOINT_REGEX  # 27, no fingers

# β governor: max PD-target step per control tick (rad). At 50 Hz this
# bounds target slew to 15 rad/s and the PD-error torque kick to β·kp.
ACTION_BOUND = 0.3

# Height profile (flat ground). Start = crouch init pelvis height; target =
# pelvis resting on ground supine (pelvis collision sphere + clearance).
START_HEIGHT = 0.95
TARGET_HEIGHT = 0.15


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat-ground scene. Ground friction nominal 1.0; DR via events."""

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

    # history_length >= 2 required by contact_force_yank.
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
    """Contact-rich task → keep contact-relevant DR ON from iter 0
    (friction, restitution, PD gains), unlike the balance line's DR-off
    stance. Body/CoM DR stays OFF — SYM is the calibrated (shod) body."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.1),
            "dynamic_friction_range": (0.4, 1.1),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    randomize_motor_strength = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.3, 0.3)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    # Small posture noise around the standing/crouch default — v1 trigger
    # states are "quiet standing". v2: init from balance-policy rollout
    # states + zero-actuation dropout (FIRM) for any-posture coverage.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class ActionsCfg:
    """27-joint delta-from-measured-position action with the β bound.

    clip applies to PROCESSED actions (post-scale), so the target step is
    hard-limited to ±ACTION_BOUND rad regardless of raw policy output.
    """

    JointPositionAction = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=BODY_JOINT_REGEX,
        scale=ACTION_BOUND,
        clip={".*": (-ACTION_BOUND, ACTION_BOUND)},
        use_zero_offset=True,
    )


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable set only — identical channels to the balance line."""

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

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged: root height + per-body contact flags stand in for the
        contact-stage indicator (Contact-Staged Stand-up: stage info to the
        critic only, actor stays deployable)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        root_height = ObsTerm(func=mdp.root_height_obs)
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
        contact_flags = ObsTerm(
            func=mdp.body_contact_flags,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        ".*_ankle_roll_link", ".*_knee_link", "pelvis",
                        # hand base links are URDF-merged into wrist_yaw —
                        # wrist_yaw IS the hand body in sim
                        "torso_link", ".*_wrist_yaw_link",
                    ],
                ),
                "threshold": 5.0,
            },
        )
        last_action = ObsTerm(func=mdp.last_action)

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Three groups (single critic for v1; group weights folded into term
    weights). Ratios follow HoST task:style:reg ≈ 2.5:1:0.1 in spirit —
    quiet_lying is deliberately the largest prize so the episode's optimum
    is 'down, supine, and STILL', not 'down fast'."""

    # ---- Task: get down, get supine, get still ----
    height_descent = RewTerm(
        func=mdp.height_descent_progress,
        weight=15.0,   # sd2: 4.0 -> 15.0. Getting DOWN is the job; it was the
                       # weakest task term while every regulariser fired.
        params={"target_height": TARGET_HEIGHT, "start_height": START_HEIGHT},
    )
    lying_orientation = RewTerm(
        func=mdp.lying_orientation_exp,
        weight=6.0,
        params={
            "target_gravity_b": [-1.0, 0.0, 0.0],   # supine: base x-axis up
            "std": 0.7,
            "gate_height": 0.5,
        },
    )
    quiet_lying = RewTerm(
        func=mdp.quiet_lying_bonus,
        weight=10.0,
        params={
            "gate_height": 0.35,
            "gate_gravity_x": -0.6,
            "std_lin": 0.10,
            "std_ang": 0.30,
            "std_jvel": 1.0,
        },
    )

    # sd2: reward a slow DELIBERATE descent, not merely the absence of a fast one.
    # descent_rate_limit alone scored "never move" == "descend perfectly".
    controlled_descent = RewTerm(
        func=mdp.controlled_descent_bonus,
        weight=8.0,
        params={"target_speed": 0.20, "std": 0.12, "min_height": 0.30},
    )
    # sd2: the COMPLETION payment — replaces an alive bonus for a one-shot task.
    # Conditional on having finished (low + supine + stopped), not on time survived.
    settled_supine = RewTerm(
        func=mdp.settled_supine_bonus,
        weight=20.0,
        params={"height_thr": 0.30, "gravity_x_thr": -0.7, "lin_thr": 0.15, "ang_thr": 0.5},
    )
    # sd2: TERMINAL PENALTY on the forbidden terminal state. This is the fix for
    # the learned-suicide exploit: faceplanting was a free exit from accumulated
    # cost, so the optimum was to reach it in 9 steps. Now it costs more than the
    # whole episode's regularisers.
    faceplant_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-200.0,
        params={"term_keys": ["faceplant"]},
    )

    # ---- Safety: slow + soft (FIRM damage triplet + SafeFall groups) ----
    descent_rate = RewTerm(
        func=mdp.descent_rate_limit,
        weight=-15.0,
        params={"max_down_speed": 0.3},
    )
    yank = RewTerm(
        func=mdp.contact_force_yank,
        weight=-2.0e-6,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
    )
    momentum = RewTerm(func=mdp.body_momentum_rate, weight=-2.0e-3)

    # Impact-force excess per body group. Thresholds sit above each group's
    # plausible resting share of bodyweight (~620 N total supine) so quiet
    # lying is free and only slams/hard leans pay.
    torso_impact = RewTerm(
        func=mdp.contact_force_above_threshold,
        weight=-0.20,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link"]),
            "threshold": 450.0,
        },
    )
    arm_impact = RewTerm(
        func=mdp.contact_force_above_threshold,
        weight=-0.10,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                # wrist_yaw carries the merged hand bodies (URDF fixed-joint merge)
                body_names=[".*_shoulder_.*_link", ".*_elbow_.*_link",
                            ".*_wrist_.*_link"],
            ),
            "threshold": 100.0,
        },
    )
    knee_hip_impact = RewTerm(
        func=mdp.contact_force_above_threshold,
        weight=-0.03,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_knee_link", ".*_hip_.*_link"]
            ),
            "threshold": 250.0,
        },
    )
    pelvis_impact = RewTerm(
        func=mdp.contact_force_above_threshold,
        weight=-0.03,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis"]),
            "threshold": 600.0,
        },
    )

    # Arms carry load only lightly: τ² scoped to arms (40/18 Nm actuators).
    arm_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_REGEX)},
    )

    # ---- Regularization (HoST group, our units) ----
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_smoothness = RewTerm(func=mdp.action_smoothness_2nd, weight=-0.05)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_REGEX)},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_REGEX)},
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.5e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_REGEX)},
    )
    joint_power = RewTerm(
        func=mdp.energy,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_REGEX)},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_REGEX)},
    )
    base_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)


@configclass
class TerminationsCfg:
    """FIRM-minimal. Deliberately ABSENT: base-height minimum (lying is the
    goal), non-foot contact (every body may touch ground)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    ballistic = DoneTerm(
        func=mdp.root_velocity_limit,
        params={"max_lin_vel": 2.5, "max_ang_vel": 6.0},
    )
    faceplant = DoneTerm(
        func=mdp.prone_orientation,
        params={"gravity_x_threshold": 0.75},
    )


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # no commands, no curriculum — single unconditional behavior

    def __post_init__(self):
        self.decimation = 4
        # Horizon deliberately longer than the descent (FIRM): ~8 s to get
        # down leaves ~12 s where quiet_lying is the only income — holding
        # the final pose IS the majority of the optimal return.
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # Whole-body ground contact → many more contact pairs than the
        # balance task; keep the doubled patch buffer.
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
