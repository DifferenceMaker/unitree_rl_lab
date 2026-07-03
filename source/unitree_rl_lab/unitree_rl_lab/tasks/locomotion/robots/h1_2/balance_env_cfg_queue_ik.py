"""Queue task for the p11 IK envelope — Balance-QIK.

Same override mechanics as balance_env_cfg_queue (QUEUE_JOB_JSON with
set_weight / set_param / add_reward / edit_raw), but the arm command is the
IK-resolved one (IKArmPoseCommand) and the phase4 wobble curriculum is
replaced by the workspace-scale ramp. Job overrides are applied AFTER the
swap, so jobs can tune the IK command via edit_raw, e.g.:

    edit_raw 'self.commands.arm_pose_command.default_pose_prob = 0.40'
    edit_raw 'self.curriculum.ik_workspace.params["scale_levels"] = (0.5, 0.75, 1.0)'

Run via a dedicated queue dir + runner instance:
    queue_runner.py --queue-dir /data/work_toms/queue_ik ... --task Unitree-H1_2-Balance-QIK
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import ARM_JOINT_REGEX, RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_queue import _apply_overrides, _load_overrides


def _swap_in_ik_command(cfg):
    """Replace the wobble arm command with the IK envelope + its curriculum."""
    cfg.commands.arm_pose_command = mdp.IKArmPoseCommandCfg(
        asset_name="robot",
        all_arm_joint_names=ARM_JOINT_REGEX,
        resampling_time_range=(2.0, 10.0),
        default_pose_prob=0.25,   # peace-time anchor (batch2 conclusion 1)
        workspace_scale=1.0,      # overridden by the curriculum from step 0
        debug_vis=False,
    )
    # phase4 drives UniformArmPoseCommand-only knobs — replace with the ramp.
    cfg.curriculum.arm_pose = None
    cfg.curriculum.ik_workspace = CurrTerm(
        func=mdp.ik_workspace_scale_curriculum,
        params={
            "command_term_name": "arm_pose_command",
            "warmup_steps": 6000,       # ~250 iter near-default targets
            "hold_steps": 12000,        # ~500 iter per level
            "scale_levels": (0.6, 0.8, 1.0),
        },
    )


@configclass
class RobotEnvCfgQueueIK(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _swap_in_ik_command(self)
        _apply_overrides(self, _load_overrides())


@configclass
class RobotPlayEnvCfgQueueIK(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _swap_in_ik_command(self)
        self.commands.arm_pose_command.debug_vis = True
        # Play at the full workspace immediately.
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
