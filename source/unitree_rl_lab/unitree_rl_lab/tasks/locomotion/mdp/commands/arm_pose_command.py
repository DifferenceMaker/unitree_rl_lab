"""Arm pose command for option (ii) Phase 2.

Phase 2 extends Phase 1 by adding a second sampling dimension:
shoulder_roll. Each episode now samples:
- One shoulder_pitch delta (applied to both arms identically)
- One shoulder_roll delta (applied mirrored — left positive, right
  negative — so both arms spread outward symmetrically)

Both held for the full episode. All other arm joints stay at default.

Bilateral roll mirroring:
- shoulder_roll has asymmetric limits: left [-0.38, 3.4] (outward
  is positive), right [-3.4, 0.38] (outward is negative)
- For bilateral symmetric spreading, sample one roll_delta_magnitude,
  then apply +delta to left and -delta to right (both spread outward)
- If sampled delta is negative (e.g., -0.3), arms try to spread inward
  (across body) but blocked by torso — joint limits clip silently
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


SHOULDER_PITCH_JOINT_REGEX = [".*_shoulder_pitch_joint"]
LEFT_SHOULDER_ROLL_REGEX = ["left_shoulder_roll_joint"]
RIGHT_SHOULDER_ROLL_REGEX = ["right_shoulder_roll_joint"]


class UniformArmPoseCommand(CommandTerm):
    """Episode-stable bilateral shoulder pitch + roll disturbance.

    Phase 2 specifics:
    - command_b stores per-env pitch_delta and roll_delta_magnitude
    - shoulder_pitch: both arms get same delta (bilateral same-direction)
    - shoulder_roll: left arm gets +delta_magnitude, right gets -delta_magnitude
      (bilateral mirrored — both arms spread outward when delta is positive,
      both spread inward when negative)
    - All other arm joints stay at default values
    - 14-dim observation vector: shows the full commanded arm pose so
      the policy can anticipate the disturbance
    """

    cfg: "UniformArmPoseCommandCfg"

    def __init__(self, cfg: "UniformArmPoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # All arm joints (for observation vector)
        self.all_arm_joint_ids, self.all_arm_joint_names = self.robot.find_joints(cfg.all_arm_joint_names)
        self.num_arm_joints = len(self.all_arm_joint_ids)

        # Shoulder pitch joints (bilateral, both arms get same delta)
        self.pitch_joint_ids, _ = self.robot.find_joints(SHOULDER_PITCH_JOINT_REGEX)

        # Shoulder roll joints — separate left and right because we
        # apply mirrored deltas
        self.left_roll_joint_ids, _ = self.robot.find_joints(LEFT_SHOULDER_ROLL_REGEX)
        self.right_roll_joint_ids, _ = self.robot.find_joints(RIGHT_SHOULDER_ROLL_REGEX)

        # Where in the 14-arm-joint observation vector are the relevant dims?
        self.pitch_indices_in_arm_obs = [
            self.all_arm_joint_ids.index(pid) for pid in self.pitch_joint_ids
        ]
        self.left_roll_indices_in_arm_obs = [
            self.all_arm_joint_ids.index(pid) for pid in self.left_roll_joint_ids
        ]
        self.right_roll_indices_in_arm_obs = [
            self.all_arm_joint_ids.index(pid) for pid in self.right_roll_joint_ids
        ]

        # Per-env current deltas
        self.pitch_delta = torch.zeros(env.num_envs, device=env.device)
        self.roll_delta = torch.zeros(env.num_envs, device=env.device)

        # 14-dim observation vector
        self.command_b = torch.zeros(env.num_envs, self.num_arm_joints, device=env.device)

        # Default joint positions (for absolute target computation)
        self.default_arm_pos = self.robot.data.default_joint_pos[:, self.all_arm_joint_ids].clone()
        self.default_pitch_pos = self.robot.data.default_joint_pos[:, self.pitch_joint_ids].clone()
        self.default_left_roll_pos = self.robot.data.default_joint_pos[:, self.left_roll_joint_ids].clone()
        self.default_right_roll_pos = self.robot.data.default_joint_pos[:, self.right_roll_joint_ids].clone()

    def __str__(self) -> str:
        return (
            f"UniformArmPoseCommand(Phase 2, bilateral pitch + mirrored roll, "
            f"pitch_amplitude={self.cfg.pitch_amplitude:.3f}, "
            f"roll_amplitude={self.cfg.roll_amplitude:.3f}, "
            f"apply_directly={self.cfg.apply_directly})"
        )

    @property
    def command(self) -> torch.Tensor:
        """The 14-dim observation: default + deltas on the relevant dims."""
        return self.default_arm_pos + self.command_b

    def _resample_command(self, env_ids):
        """Resample pitch and roll deltas for the given envs (episode reset)."""
        n = len(env_ids)
        if n == 0:
            return

        # Sample pitch delta uniformly in [-pitch_amplitude, +pitch_amplitude]
        new_pitch = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.pitch_amplitude
        self.pitch_delta[env_ids] = new_pitch

        # Sample roll delta uniformly in [-roll_amplitude, +roll_amplitude]
        # Positive = both arms spread outward, negative = both try inward (limit-blocked)
        new_roll = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.roll_amplitude
        self.roll_delta[env_ids] = new_roll

        # Update the 14-dim observation vector for these envs
        self.command_b[env_ids] = 0.0
        for idx in self.pitch_indices_in_arm_obs:
            self.command_b[env_ids, idx] = new_pitch
        # Left roll = +delta (outward direction is positive for left)
        for idx in self.left_roll_indices_in_arm_obs:
            self.command_b[env_ids, idx] = new_roll
        # Right roll = -delta (outward direction is negative for right)
        for idx in self.right_roll_indices_in_arm_obs:
            self.command_b[env_ids, idx] = -new_roll

    def _update_command(self):
        """Apply targets to sim each step. No resampling — held for episode."""
        if not self.cfg.apply_directly:
            return

        # Pitch: both arms get same target = default + pitch_delta
        pitch_target = self.default_pitch_pos + self.pitch_delta.unsqueeze(1)
        self.robot.set_joint_position_target(
            pitch_target,
            joint_ids=self.pitch_joint_ids,
        )

        # Left roll: default + delta (positive delta = outward for left arm)
        left_roll_target = self.default_left_roll_pos + self.roll_delta.unsqueeze(1)
        self.robot.set_joint_position_target(
            left_roll_target,
            joint_ids=self.left_roll_joint_ids,
        )

        # Right roll: default - delta (negative direction = outward for right arm)
        right_roll_target = self.default_right_roll_pos - self.roll_delta.unsqueeze(1)
        self.robot.set_joint_position_target(
            right_roll_target,
            joint_ids=self.right_roll_joint_ids,
        )

    def _update_metrics(self):
        pass


@configclass
class UniformArmPoseCommandCfg(CommandTermCfg):
    """Cfg for UniformArmPoseCommand (Phase 2)."""

    class_type: type = UniformArmPoseCommand

    asset_name: str = MISSING
    """Name of the articulation in the scene to command."""

    all_arm_joint_names: list[str] = MISSING
    """All 14 arm joint names — for the observation vector."""

    pitch_amplitude: float = 0.0
    """Max delta from default shoulder_pitch (radians).
    Phase 2 keeps this fixed at 1.5 across the curriculum."""

    roll_amplitude: float = 0.0
    """Max magnitude of mirrored shoulder_roll delta (radians).
    Overridden by curriculum. Starts at 0, ramps up over Phase 2 levels."""

    apply_directly: bool = True
    """If True (option ii), command writes joint targets to sim directly."""

    # Not used since we resample at reset, parent class requires it
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    debug_vis: bool = False
