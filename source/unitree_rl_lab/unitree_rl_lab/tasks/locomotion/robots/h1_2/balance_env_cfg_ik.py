"""IK-envelope variant of the balance env (p11).

Swaps the joint-angle arm command (UniformArmPoseCommand) for the
IK-resolved one (IKArmPoseCommand): Cartesian hand targets in the torso
frame, differential-IK joint targets, 2-10 s mid-episode resampling, and a
peace-time (default pose) fraction. The old arm_pose curriculum is removed —
its knobs (yaw/decouple/overhead amplitudes) don't exist on the IK command;
the IK curriculum knob is `workspace_scale` (ramped by queue jobs later).

Everything else inherits from balance_env_cfg.RobotEnvCfg.
"""

from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import ARM_JOINT_REGEX, RobotEnvCfg


@configclass
class RobotEnvCfgIK(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands.arm_pose_command = mdp.IKArmPoseCommandCfg(
            asset_name="robot",
            all_arm_joint_names=ARM_JOINT_REGEX,
            resampling_time_range=(2.0, 10.0),
            default_pose_prob=0.25,
            workspace_scale=1.0,
            debug_vis=False,
        )

        # The phase4 curriculum drives UniformArmPoseCommand-only knobs.
        self.curriculum.arm_pose = None


@configclass
class RobotPlayEnvCfgIK(RobotEnvCfgIK):
    """Play-time: few envs, visible hand-target dots, max pushes (as in the
    base play cfg, minus the arm_pose curriculum forcing)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32

        self.commands.arm_pose_command.debug_vis = True

        # Force max impulse push velocity (skip ramp) — mirrors RobotPlayEnvCfg.
        self.curriculum.push_velocity.params["warmup_steps"] = 0
        self.curriculum.push_velocity.params["levels"] = (0.5,) * 2
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5),
        }

        # Force max sustained push (skip warmup, top level).
        self.curriculum.sustained_push.params["warmup_steps"] = 0
        self.curriculum.sustained_push.params["levels"] = (
            ((0.0, 30.0), (2.0, 3.5)),
        ) * 4
        self.events.sustained_push_apply.params["force_magnitude_range"] = (0.0, 30.0)
        self.events.sustained_push_apply.params["duration_range_s"] = (2.0, 4.0)
