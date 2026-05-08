"""Arm pose command for tracking-curriculum training.

Generates random arm joint position targets per episode, sampled within
±amplitude rad of the default joint position. Both amplitude and
resample_period_s are intended to be overridden each step by the
arm_amplitude_curriculum term.

The command is designed to work alongside the existing UniformLevelVelocityCommand
without conflict. Where velocity command tells the robot 'how fast to move
the base,' arm_pose_command tells it 'where the arms should be right now.'
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


class UniformArmPoseCommand(CommandTerm):
    """Sample random arm joint position targets per resample period.

    For each environment, sample target = default_pose + uniform(-amplitude, amplitude)
    on each commanded joint. Resample every resample_period_s seconds (per env).
    Both amplitude and resample_period_s are read from cfg every step, so the
    curriculum can override them dynamically.
    """

    cfg: "UniformArmPoseCommandCfg"

    def __init__(self, cfg: "UniformArmPoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        # Resolve joint indices from the regex/list given in cfg.joint_names
        self.joint_ids, self.joint_names = self.robot.find_joints(cfg.joint_names)
        self.num_joints = len(self.joint_ids)

        # Per-env time since last resample (in seconds)
        self.time_since_resample = torch.zeros(env.num_envs, device=env.device)
        # Per-env current arm pose target (relative to default, ready to add)
        self.command_b = torch.zeros(env.num_envs, self.num_joints, device=env.device)

        # Default joint positions for the arm joints, broadcast for sampling
        self.default_pos = self.robot.data.default_joint_pos[:, self.joint_ids].clone()

    def __str__(self) -> str:
        return (
            f"UniformArmPoseCommand({self.num_joints} joints, "
            f"amplitude={self.cfg.amplitude:.3f} rad, "
            f"resample_period={self.cfg.resample_period_s:.2f}s)"
        )

    @property
    def command(self) -> torch.Tensor:
        """The currently commanded arm joint pose (absolute, world frame).

        Returns shape (num_envs, num_joints), in radians. This is what the
        policy observes and what the reward should compare actual joint
        positions against.
        """
        return self.default_pos + self.command_b

    def _resample_command(self, env_ids: "Sequence[int]"):
        """Resample arm targets for the given environments."""
        n = len(env_ids)
        if n == 0:
            return
        # Uniform in [-amplitude, +amplitude]
        new_targets = (
            torch.rand(n, self.num_joints, device=self.device) * 2.0 - 1.0
        ) * self.cfg.amplitude
        self.command_b[env_ids] = new_targets
        self.time_since_resample[env_ids] = 0.0

    def _update_command(self):
        """Called every env step. Increment per-env timer and resample where elapsed."""
        self.time_since_resample += self._env.step_dt
        # Find envs where the timer has elapsed
        needs_resample = (self.time_since_resample >= self.cfg.resample_period_s).nonzero(as_tuple=False).flatten()
        if len(needs_resample) > 0:
            self._resample_command(needs_resample)

    def _update_metrics(self):
        """No specific metrics to track for this command."""
        pass


@configclass
class UniformArmPoseCommandCfg(CommandTermCfg):
    """Cfg for UniformArmPoseCommand."""

    class_type: type = UniformArmPoseCommand

    asset_name: str = MISSING
    """Name of the articulation in the scene to command."""

    joint_names: list[str] = MISSING
    """Regex/list of joint names to command (e.g., arm joints)."""

    amplitude: float = 0.05
    """Maximum deviation from default joint position (radians).
    Sample uniform in [-amplitude, +amplitude]. Overridden by curriculum."""

    resample_period_s: float = 4.0
    """How often to resample new targets (seconds, per env). Overridden by curriculum."""

    # Inherit from CommandTermCfg — set defaults appropriate for this term
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    """Not used by this command (we manage resampling internally), but
    required by CommandTermCfg parent class."""

    debug_vis: bool = False
