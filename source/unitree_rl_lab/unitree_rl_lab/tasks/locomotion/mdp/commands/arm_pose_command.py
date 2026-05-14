"""Arm pose command for option (ii) Phase 3.

Phase 3 extends Phase 2 by adding a third sampling dimension:
elbow_pitch. Each episode now samples:
- One shoulder_pitch delta (applied to both arms identically)
- One shoulder_roll delta magnitude (mirrored: left +, right - for outward
  spreading)
- One elbow_pitch delta (applied to both arms identically — bilateral bend)

All held for the full episode. shoulder_yaw and wrist joints stay at default.

H1-2 elbow_pitch joint:
- Default: 0.3 rad (~17 deg, slight resting bend)
- Limits: [-0.95, 3.18]
- Positive direction = curl forearm toward shoulder (fold arm)
- Negative direction = extend past straight
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
ELBOW_PITCH_JOINT_REGEX = [".*_elbow_pitch_joint"]


class UniformArmPoseCommand(CommandTerm):
    """Episode-stable bilateral pitch + mirrored roll + bilateral elbow.

    Phase 3 specifics:
    - 3 sampled values per episode: pitch_delta, roll_delta_mag, elbow_delta
    - pitch: both arms get same delta
    - roll: mirrored (left +delta, right -delta) for symmetric outward spread
    - elbow: both arms get same delta (bilateral bend)
    - shoulder_yaw and wrist joints stay at default values
    """

    cfg: "UniformArmPoseCommandCfg"

    def __init__(self, cfg: "UniformArmPoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # All arm joints (for observation vector)
        self.all_arm_joint_ids, self.all_arm_joint_names = self.robot.find_joints(cfg.all_arm_joint_names)
        self.num_arm_joints = len(self.all_arm_joint_ids)

        # Joint subsets we're commanding
        self.pitch_joint_ids, _ = self.robot.find_joints(SHOULDER_PITCH_JOINT_REGEX)
        self.left_roll_joint_ids, _ = self.robot.find_joints(LEFT_SHOULDER_ROLL_REGEX)
        self.right_roll_joint_ids, _ = self.robot.find_joints(RIGHT_SHOULDER_ROLL_REGEX)
        self.elbow_joint_ids, _ = self.robot.find_joints(ELBOW_PITCH_JOINT_REGEX)

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
        self.elbow_indices_in_arm_obs = [
            self.all_arm_joint_ids.index(pid) for pid in self.elbow_joint_ids
        ]

        # Per-env current deltas
        self.pitch_delta = torch.zeros(env.num_envs, device=env.device)
        self.roll_delta = torch.zeros(env.num_envs, device=env.device)
        self.elbow_delta = torch.zeros(env.num_envs, device=env.device)

        # 14-dim observation vector
        self.command_b = torch.zeros(env.num_envs, self.num_arm_joints, device=env.device)

        # Default joint positions
        self.default_arm_pos = self.robot.data.default_joint_pos[:, self.all_arm_joint_ids].clone()
        self.default_pitch_pos = self.robot.data.default_joint_pos[:, self.pitch_joint_ids].clone()
        self.default_left_roll_pos = self.robot.data.default_joint_pos[:, self.left_roll_joint_ids].clone()
        self.default_right_roll_pos = self.robot.data.default_joint_pos[:, self.right_roll_joint_ids].clone()
        self.default_elbow_pos = self.robot.data.default_joint_pos[:, self.elbow_joint_ids].clone()

    def __str__(self) -> str:
        return (
            f"UniformArmPoseCommand(Phase 3, bilateral pitch + mirrored roll "
            f"+ bilateral elbow, pitch_amp={self.cfg.pitch_amplitude:.3f}, "
            f"roll_amp={self.cfg.roll_amplitude:.3f}, "
            f"elbow_amp={self.cfg.elbow_amplitude:.3f}, "
            f"apply_directly={self.cfg.apply_directly})"
        )

    @property
    def command(self) -> torch.Tensor:
        """14-dim observation: default + deltas on the relevant dims."""
        return self.default_arm_pos + self.command_b

    def _resample_command(self, env_ids):
        """Resample pitch, roll, elbow deltas at episode reset."""
        n = len(env_ids)
        if n == 0:
            return

        # Sample uniformly in [-amplitude, +amplitude] for each dimension
        new_pitch = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.pitch_amplitude
        new_roll = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.roll_amplitude
        new_elbow = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.elbow_amplitude

        self.pitch_delta[env_ids] = new_pitch
        self.roll_delta[env_ids] = new_roll
        self.elbow_delta[env_ids] = new_elbow

        # Update 14-dim observation vector for these envs
        self.command_b[env_ids] = 0.0
        for idx in self.pitch_indices_in_arm_obs:
            self.command_b[env_ids, idx] = new_pitch
        for idx in self.left_roll_indices_in_arm_obs:
            self.command_b[env_ids, idx] = new_roll
        for idx in self.right_roll_indices_in_arm_obs:
            self.command_b[env_ids, idx] = -new_roll
        for idx in self.elbow_indices_in_arm_obs:
            self.command_b[env_ids, idx] = new_elbow

    def _update_command(self):
        """Apply targets to sim each step (held for episode)."""
        if not self.cfg.apply_directly:
            return

        # Pitch: both arms get same target
        pitch_target = self.default_pitch_pos + self.pitch_delta.unsqueeze(1)
        self.robot.set_joint_position_target(pitch_target, joint_ids=self.pitch_joint_ids)

        # Left roll: default + delta (positive delta = outward for left arm)
        left_roll_target = self.default_left_roll_pos + self.roll_delta.unsqueeze(1)
        self.robot.set_joint_position_target(left_roll_target, joint_ids=self.left_roll_joint_ids)

        # Right roll: default - delta (mirrored: negative direction = outward for right)
        right_roll_target = self.default_right_roll_pos - self.roll_delta.unsqueeze(1)
        self.robot.set_joint_position_target(right_roll_target, joint_ids=self.right_roll_joint_ids)

        # Elbow: both arms get same target (bilateral bend)
        elbow_target = self.default_elbow_pos + self.elbow_delta.unsqueeze(1)
        self.robot.set_joint_position_target(elbow_target, joint_ids=self.elbow_joint_ids)

    def _update_metrics(self):
        pass


@configclass
class UniformArmPoseCommandCfg(CommandTermCfg):
    """Cfg for UniformArmPoseCommand (Phase 3)."""

    class_type: type = UniformArmPoseCommand

    asset_name: str = MISSING
    all_arm_joint_names: list[str] = MISSING

    pitch_amplitude: float = 0.0
    """Max delta from default shoulder_pitch (radians).
    Phase 3 keeps this at Phase 2 max (1.5)."""

    roll_amplitude: float = 0.0
    """Max magnitude of mirrored shoulder_roll delta (radians).
    Phase 3 keeps this at Phase 2 max (1.0)."""

    elbow_amplitude: float = 0.0
    """Max delta from default elbow_pitch (radians).
    Set by curriculum. Phase 3 levels: 0.0, 0.5, 1.0, 1.5."""

    apply_directly: bool = True
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    debug_vis: bool = False
