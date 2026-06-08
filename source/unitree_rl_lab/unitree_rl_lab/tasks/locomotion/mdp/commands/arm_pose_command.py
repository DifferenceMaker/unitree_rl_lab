"""Arm pose command for option (ii) Phase 4.

Phase 4 extends Phase 3 by adding sinusoidal wobble on top of the
held pose. Each step the target becomes:

    target = held_pose + wobble_amplitude * sin(2*pi * freq * t + phase)

Each episode samples:
- pitch_delta (bilateral, held)         [Phase 1]
- roll_delta_magnitude (mirrored, held) [Phase 2]
- elbow_delta (bilateral, held)         [Phase 3]
- per-joint phase offset for wobble     [Phase 4]

Wobble settings:
- frequency: ~2 Hz (configurable, default 2.0)
- amplitude: from curriculum
- per-joint independent phase offsets to avoid all joints wobbling in sync
- wobble applied to: shoulder_pitch (both arms), shoulder_roll (left & right),
  elbow_pitch (both arms) — 4 wobble channels total

Mimics manipulation micro-motions (gripper movement, hand articulation,
small mass shifts during object manipulation).
"""

from __future__ import annotations

import math
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
    """Episode-stable held pose + sinusoidal wobble per joint.

    v4.1: per-episode per-env amplitude sampling.
    
    Each episode reset:
      - pitch_amp_episode  ~ U(0, pitch_amplitude_max)
      - roll_amp_episode   ~ U(0, roll_amplitude_max)
      - elbow_amp_episode  ~ U(0, elbow_amplitude_max)
      - wobble_amp_episode ~ U(0, wobble_amplitude_max)
      - 4 wobble phase offsets uniform in [0, 2pi]
      - Held deltas sampled within that episode's amplitudes
    
    This ensures coverage of:
      - "stand still, no arms" (amplitudes near 0)
      - "held pose, no wobble" (held amps > 0, wobble amp near 0)
      - "wobble only" (held amps near 0, wobble amp > 0)
      - "max disturbance" (all amps near max)
    
    The policy must learn ALL of these simultaneously, preparing it for the
    full sim2real distribution (Idle/Mild/Training modes on the C++ side).
    """

    cfg: "UniformArmPoseCommandCfg"

    def __init__(self, cfg: "UniformArmPoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self.all_arm_joint_ids, self.all_arm_joint_names = self.robot.find_joints(cfg.all_arm_joint_names)
        self.num_arm_joints = len(self.all_arm_joint_ids)

        self.pitch_joint_ids, _ = self.robot.find_joints(SHOULDER_PITCH_JOINT_REGEX)
        self.left_roll_joint_ids, _ = self.robot.find_joints(LEFT_SHOULDER_ROLL_REGEX)
        self.right_roll_joint_ids, _ = self.robot.find_joints(RIGHT_SHOULDER_ROLL_REGEX)
        self.elbow_joint_ids, _ = self.robot.find_joints(ELBOW_PITCH_JOINT_REGEX)

        # Indices in 14-arm-joint observation vector
        self.pitch_indices_in_arm_obs = [self.all_arm_joint_ids.index(pid) for pid in self.pitch_joint_ids]
        self.left_roll_indices_in_arm_obs = [self.all_arm_joint_ids.index(pid) for pid in self.left_roll_joint_ids]
        self.right_roll_indices_in_arm_obs = [self.all_arm_joint_ids.index(pid) for pid in self.right_roll_joint_ids]
        self.elbow_indices_in_arm_obs = [self.all_arm_joint_ids.index(pid) for pid in self.elbow_joint_ids]

        # Per-env held deltas
        self.pitch_delta = torch.zeros(env.num_envs, device=env.device)
        self.roll_delta = torch.zeros(env.num_envs, device=env.device)
        self.elbow_delta = torch.zeros(env.num_envs, device=env.device)

        # Per-env per-channel phase offsets for wobble (4 channels:
        # pitch, left_roll, right_roll, elbow)
        self.wobble_phase = torch.zeros(env.num_envs, 4, device=env.device)

        # Per-env time tracking for wobble (reset on episode reset)
        self.t = torch.zeros(env.num_envs, device=env.device)

        # Observation vector (held pose; wobble not in obs to avoid distraction)
        self.command_b = torch.zeros(env.num_envs, self.num_arm_joints, device=env.device)

        # Default positions
        self.default_arm_pos = self.robot.data.default_joint_pos[:, self.all_arm_joint_ids].clone()
        self.default_pitch_pos = self.robot.data.default_joint_pos[:, self.pitch_joint_ids].clone()
        self.default_left_roll_pos = self.robot.data.default_joint_pos[:, self.left_roll_joint_ids].clone()
        self.default_right_roll_pos = self.robot.data.default_joint_pos[:, self.right_roll_joint_ids].clone()
        self.default_elbow_pos = self.robot.data.default_joint_pos[:, self.elbow_joint_ids].clone()

        # Per-env per-episode amplitude samples (NEW for v4.1)
        self.pitch_amp_episode = torch.zeros(env.num_envs, device=env.device)
        self.roll_amp_episode = torch.zeros(env.num_envs, device=env.device)
        self.elbow_amp_episode = torch.zeros(env.num_envs, device=env.device)
        self.wobble_amp_episode = torch.zeros(env.num_envs, device=env.device)

        # Step dt for time integration
        self.dt = env.step_dt

    def __str__(self) -> str:
        return (
            f"UniformArmPoseCommand(Phase 4, held pose + wobble, "
            f"pitch_amp={self.cfg.pitch_amplitude:.3f}, "
            f"roll_amp={self.cfg.roll_amplitude:.3f}, "
            f"elbow_amp={self.cfg.elbow_amplitude:.3f}, "
            f"wobble_amp={self.cfg.wobble_amplitude:.3f}, "
            f"wobble_freq={self.cfg.wobble_frequency:.2f}Hz)"
        )

    @property
    def command(self) -> torch.Tensor:
        """14-dim observation: held pose only (no wobble in obs).

        Policy observes the commanded held pose. The wobble is treated as
        a small disturbance the policy must reactively handle, not predict.
        """
        return self.default_arm_pos + self.command_b

    def _resample_command(self, env_ids):
        """Resample episode amplitudes, held deltas, and wobble phase at episode reset."""
        n = len(env_ids)
        if n == 0:
            return

        # v4.1: per-episode per-env amplitude sampling — uniformly from [0, max].
        # This means each episode can be anywhere from "no arm motion" to "max disturbance",
        # including mixed configs like "held offset but no wobble" or "wobble but no offset".
        self.pitch_amp_episode[env_ids] = torch.rand(n, device=self.device) * self.cfg.pitch_amplitude
        self.roll_amp_episode[env_ids] = torch.rand(n, device=self.device) * self.cfg.roll_amplitude
        self.elbow_amp_episode[env_ids] = torch.rand(n, device=self.device) * self.cfg.elbow_amplitude
        self.wobble_amp_episode[env_ids] = torch.rand(n, device=self.device) * self.cfg.wobble_amplitude

        # Sample held deltas within this episode's amplitudes
        new_pitch = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.pitch_amp_episode[env_ids]
        new_roll = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.roll_amp_episode[env_ids]
        new_elbow = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.elbow_amp_episode[env_ids]

        self.pitch_delta[env_ids] = new_pitch
        self.roll_delta[env_ids] = new_roll
        self.elbow_delta[env_ids] = new_elbow

        # Sample wobble phase offsets uniformly in [0, 2*pi]
        self.wobble_phase[env_ids] = torch.rand(n, 4, device=self.device) * (2.0 * math.pi)

        # Reset time
        self.t[env_ids] = 0.0

        # Update observation vector (held pose only)
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
        """Apply held pose + wobble each step."""
        if not self.cfg.apply_directly:
            return

        # Advance time
        self.t += self.dt
        omega_t = 2.0 * math.pi * self.cfg.wobble_frequency * self.t

        # v4.1: per-env wobble amplitude (was scalar self.cfg.wobble_amplitude)
        # wobble_amp_episode shape: (num_envs,) — broadcast to (num_envs, 1) below
        wobble = self.wobble_amp_episode.unsqueeze(1) * torch.sin(
            omega_t.unsqueeze(1) + self.wobble_phase
        )
        # wobble shape: (num_envs, 4) — channels: pitch, left_roll, right_roll, elbow

        # Pitch: held delta + wobble[:, 0]
        pitch_target = self.default_pitch_pos + (self.pitch_delta + wobble[:, 0]).unsqueeze(1)
        self.robot.set_joint_position_target(pitch_target, joint_ids=self.pitch_joint_ids)

        # Left roll: held + wobble[:, 1], CLAMPED to +/-0.8 (p7_1b self-collision guard)
        left_roll_offset = torch.clamp(self.roll_delta + wobble[:, 1], -0.8, 0.8)
        left_roll_target = self.default_left_roll_pos + left_roll_offset.unsqueeze(1)
        self.robot.set_joint_position_target(left_roll_target, joint_ids=self.left_roll_joint_ids)

        # Right roll: mirrored held + wobble[:, 2]
        right_roll_offset = torch.clamp(-self.roll_delta + wobble[:, 2], -0.8, 0.8)  # p7_1b clamp
        right_roll_target = self.default_right_roll_pos + right_roll_offset.unsqueeze(1)
        self.robot.set_joint_position_target(right_roll_target, joint_ids=self.right_roll_joint_ids)

        # Elbow: held + wobble[:, 3]
        elbow_target = self.default_elbow_pos + (self.elbow_delta + wobble[:, 3]).unsqueeze(1)
        self.robot.set_joint_position_target(elbow_target, joint_ids=self.elbow_joint_ids)

    def _update_metrics(self):
        pass


@configclass
class UniformArmPoseCommandCfg(CommandTermCfg):
    """Cfg for UniformArmPoseCommand (Phase 4)."""

    class_type: type = UniformArmPoseCommand

    asset_name: str = MISSING
    all_arm_joint_names: list[str] = MISSING

    pitch_amplitude: float = 0.0
    """Max delta from default shoulder_pitch (radians). Phase 4: fixed at 1.5."""

    roll_amplitude: float = 0.0
    """Max magnitude of mirrored shoulder_roll delta (radians). Phase 4: fixed at 1.0."""

    elbow_amplitude: float = 0.0
    """Max delta from default elbow_pitch (radians). Phase 4: fixed at 1.5."""

    wobble_amplitude: float = 0.0
    """Sinusoidal wobble amplitude (radians) applied on top of held pose.
    Set by curriculum. Phase 4 levels: 0.0, 0.05, 0.10, 0.15."""

    wobble_frequency: float = 2.0
    """Wobble frequency in Hz."""

    apply_directly: bool = True
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    debug_vis: bool = False
