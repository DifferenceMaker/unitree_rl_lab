"""Arm pose command for option (ii) Phase 1.

Phase 1 design:
- Bilateral shoulder pitch sweep (both arms move together)
- Sample target ONCE at episode reset, hold for whole episode
- Apply target directly to articulation each step (bypasses policy)
- Other arm joints (shoulder_roll, shoulder_yaw, elbow, wrist) stay at default

Curriculum controls the sample amplitude. amplitude=0 means no disturbance
(arms at default). amplitude=1.0 means bilateral pitch can swing up to ±1.0 rad
from default — large sweep but within joint limits.

This is the option (ii) mechanism: arms move as an external disturbance,
policy only controls legs+torso to balance against the motion.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Joint name patterns for the two arms' shoulder_pitch joints.
# Used to find the action subset to manipulate.
SHOULDER_PITCH_JOINT_REGEX = [".*_shoulder_pitch_joint"]


class UniformArmPoseCommand(CommandTerm):
    """Episode-stable bilateral shoulder pitch disturbance.

    Phase 1 specifics:
    - command_b stores a single per-env value (the bilateral pitch delta).
      Both shoulder_pitch joints get the same value.
    - All other arm joints stay at their default values (held by simulator
      after we don't drive them).
    - command (the observable) is the full 14-arm-joint vector with the
      pitch delta applied bilaterally and zeros for other arm joints.
      Policy observes this vector.
    - Sample happens at episode reset only. _update_command writes targets
      to sim each step but doesn't resample.

    The 14-dim observation gives the policy a structured view: "left
    shoulder pitch is X, right shoulder pitch is X, everything else 0
    means default." Future phases can fill in more dims.
    """

    cfg: "UniformArmPoseCommandCfg"

    def __init__(self, cfg: "UniformArmPoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # All arm joints (for the observation vector)
        self.all_arm_joint_ids, self.all_arm_joint_names = self.robot.find_joints(cfg.all_arm_joint_names)
        self.num_arm_joints = len(self.all_arm_joint_ids)

        # Shoulder pitch joints (the subset we're disturbing in Phase 1)
        self.pitch_joint_ids, _ = self.robot.find_joints(SHOULDER_PITCH_JOINT_REGEX)
        # Where in the 14-arm-joint observation vector are the pitch dims?
        self.pitch_indices_in_arm_obs = []
        for pid in self.pitch_joint_ids:
            if pid in self.all_arm_joint_ids:
                self.pitch_indices_in_arm_obs.append(self.all_arm_joint_ids.index(pid))

        # Per-env current pitch delta (single value, applied bilaterally)
        self.pitch_delta = torch.zeros(env.num_envs, device=env.device)

        # The full 14-dim observation vector (default+delta on pitch dims,
        # zero elsewhere). Updated whenever pitch_delta is resampled.
        self.command_b = torch.zeros(env.num_envs, self.num_arm_joints, device=env.device)

        # Default positions for ALL arm joints (for absolute target calc)
        self.default_arm_pos = self.robot.data.default_joint_pos[:, self.all_arm_joint_ids].clone()
        # Default positions for just the pitch joints (for sim write)
        self.default_pitch_pos = self.robot.data.default_joint_pos[:, self.pitch_joint_ids].clone()

    def __str__(self) -> str:
        return (
            f"UniformArmPoseCommand(Phase 1, bilateral shoulder_pitch, "
            f"amplitude={self.cfg.amplitude:.3f} rad, "
            f"apply_directly={self.cfg.apply_directly})"
        )

    @property
    def command(self) -> torch.Tensor:
        """The 14-dim arm pose observation: default pose with pitch deltas applied.

        Shape: (num_envs, 14). Default values for non-pitch joints (zero
        delta = at-default). Pitch dims show the commanded delta from
        default. Policy observes this.
        """
        # Build the absolute target vector: default + delta only on pitch dims
        return self.default_arm_pos + self.command_b

    def _resample_command(self, env_ids):
        """Resample bilateral pitch delta for the given envs.

        Called at episode reset (mode='reset' in events). Samples uniformly
        in [-amplitude, +amplitude].
        """
        n = len(env_ids)
        if n == 0:
            return

        # Sample one value per env, uniformly in [-amplitude, +amplitude]
        new_delta = (
            torch.rand(n, device=self.device) * 2.0 - 1.0
        ) * self.cfg.amplitude

        self.pitch_delta[env_ids] = new_delta

        # Update the 14-dim observation vector: zero everywhere except
        # the pitch dims, which get the delta
        self.command_b[env_ids] = 0.0
        for idx in self.pitch_indices_in_arm_obs:
            self.command_b[env_ids, idx] = new_delta

    def _update_command(self):
        """Called every env step. NO resampling here — pitch is held for
        the episode. We just apply the target to the simulator each step.
        """
        if not self.cfg.apply_directly:
            return

        # Apply: shoulder_pitch_target = default + delta, both arms
        target = self.default_pitch_pos + self.pitch_delta.unsqueeze(1)
        # target shape: (num_envs, num_pitch_joints), num_pitch_joints == 2

        self.robot.set_joint_position_target(
            target,
            joint_ids=self.pitch_joint_ids,
        )

    def _update_metrics(self):
        """No specific metrics needed for Phase 1."""
        pass


@configclass
class UniformArmPoseCommandCfg(CommandTermCfg):
    """Cfg for UniformArmPoseCommand (Phase 1)."""

    class_type: type = UniformArmPoseCommand

    asset_name: str = MISSING
    """Name of the articulation in the scene to command."""

    all_arm_joint_names: list[str] = MISSING
    """All 14 arm joint names — used for the observation vector."""

    amplitude: float = 0.0
    """Max delta from default shoulder_pitch (radians).
    Overridden by curriculum. amplitude=0 means no disturbance."""

    apply_directly: bool = True
    """If True (option ii), command writes joint targets to sim directly.
    If False (option i style), command is observation-only; policy must
    drive arms via its action output."""

    # Not used since we resample at reset, but parent class requires it
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    debug_vis: bool = False
