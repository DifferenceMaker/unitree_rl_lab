from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def push_velocity_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: "Sequence[int]",
    event_term_name: str = "push_robot",
    warmup_steps: int = 6000,
    hold_steps: int = 5000,
    levels: tuple = (0.30, 0.51, 0.72, 0.94, 1.15, 1.36, 1.57, 1.79, 2.00),
) -> "torch.Tensor":
    """Stepwise push velocity curriculum.
 
    Holds at levels[0] for `warmup_steps`, then advances one level every
    `hold_steps`. Once the highest level is reached, stays there.
 
    All thresholds are in env.common_step_counter ticks.
    With num_steps_per_env=24, 1 PPO iteration = 24 ticks.
    Defaults: 6000 warmup steps (~250 iters), 5000 hold steps per level
    (~208 iters/level), 9 levels = ~1900 iters total to reach 2.0 m/s.
 
    Returns the current push velocity (x-axis upper bound) for logging.
    """
    step = env.common_step_counter
 
    if step < warmup_steps:
        target_vel = float(levels[0])
    else:
        # How many holds have elapsed since warmup ended
        levels_advanced = (step - warmup_steps) // hold_steps
        level_idx = min(int(levels_advanced), len(levels) - 1)
        target_vel = float(levels[level_idx])
 
    event_term = env.event_manager.get_term_cfg(event_term_name)
    event_term.params["velocity_range"] = {
        "x": (-target_vel, target_vel),
        "y": (-target_vel, target_vel),
    }
 
    return torch.tensor(target_vel, device=env.device)
 
# ============================================================
# ADDITIONS to curriculums.py for arm curriculum (option I).
# Append these to the existing curriculums.py — do NOT replace the file.
# ============================================================


def arm_amplitude_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: "Sequence[int]",
    command_term_name: str = "arm_pose_command",
    warmup_steps: int = 6000,
    hold_steps: int = 8000,
    amplitude_levels: tuple = (0.05, 0.10, 0.20, 0.35, 0.50, 0.70),
    resample_period_levels: tuple = (4.0, 4.0, 3.0, 2.0, 1.5, 1.0),
) -> "torch.Tensor":
    """Stepwise arm amplitude + resample period curriculum.

    Holds at level 0 for `warmup_steps` (~250 iters at 4096 envs with
    24 steps/iter), then advances one level every `hold_steps`.

    Each level configures both:
      - amplitude: max deviation in radians from default joint position
        (e.g., 0.05 ~ 3 degrees, 0.70 ~ 40 degrees)
      - resample_period_s: how often the target changes (lower = more
        continuous motion, higher = more held poses)

    The two arrays must be the same length. Levels progress together —
    higher amplitude pairs with shorter resample period, simulating
    increasingly demanding arm motion.

    Returns the current amplitude (rad) for logging.
    """
    if len(amplitude_levels) != len(resample_period_levels):
        raise ValueError(
            f"amplitude_levels and resample_period_levels must have the "
            f"same length. Got {len(amplitude_levels)} and "
            f"{len(resample_period_levels)}."
        )

    step = env.common_step_counter

    if step < warmup_steps:
        level_idx = 0
    else:
        levels_advanced = (step - warmup_steps) // hold_steps
        level_idx = min(int(levels_advanced), len(amplitude_levels) - 1)

    target_amplitude = float(amplitude_levels[level_idx])
    target_period = float(resample_period_levels[level_idx])

    # Update the command term's cfg in place. The command term reads these
    # values every step in _update_command and _resample_command.
    command_term = env.command_manager.get_term(command_term_name)
    command_term.cfg.amplitude = target_amplitude
    command_term.cfg.resample_period_s = target_period

    return torch.tensor(target_amplitude, device=env.device)


"""Curriculum for option (ii) Phase 1.

Append to mdp/curriculums.py.

Replaces arm_amplitude_curriculum from option (i). Different mechanism
(amplitude defines a static-per-episode disturbance, not a moving target).
Same structural pattern: stepwise level advancement based on env_step
counter.
"""
def arm_pose_curriculum_phase1(
    env: "ManagerBasedRLEnv",
    env_ids: "Sequence[int]",
    command_term_name: str = "arm_pose_command",
    warmup_steps: int = 6000,
    hold_steps: int = 24000,
    amplitude_levels: tuple = (0.0, 0.5, 1.0, 1.5),
) -> "torch.Tensor":
    """Phase 1 curriculum: bilateral shoulder_pitch amplitude ramps over training.

    Levels (Phase 1 — pitch only):
        0: amplitude=0.0  — arms at default (warmstart sanity check)
        1: amplitude=0.5  — small forward/backward pitch (~30 degrees)
        2: amplitude=1.0  — medium pitch (~60 degrees)
        3: amplitude=1.5  — large pitch — may hit upper joint limit at +0.4+1.5=1.9

    Default warmup_steps=6000 (~250 iters at 4096 envs).
    Default hold_steps=75000 (~3125 iters at 4096 envs) per level — generous
    convergence time per level, per user preference.

    Returns the current amplitude (rad) for logging.
    """
    step = env.common_step_counter

    if step < warmup_steps:
        level_idx = 0
    else:
        levels_advanced = (step - warmup_steps) // hold_steps
        level_idx = min(int(levels_advanced), len(amplitude_levels) - 1)

    target_amplitude = float(amplitude_levels[level_idx])

    # Update the command term's cfg in place. _resample_command reads this
    # at episode reset.
    command_term = env.command_manager.get_term(command_term_name)
    command_term.cfg.amplitude = target_amplitude

    return torch.tensor(target_amplitude, device=env.device)
"""Curriculum for option (ii) Phase 2.

Append to mdp/curriculums.py.

Phase 2 keeps shoulder_pitch amplitude fixed (at Phase 1 final value)
and ramps shoulder_roll amplitude across levels. Same stepwise pattern
as Phase 1, same iter-per-level math.
"""


def arm_pose_curriculum_phase2(
    env: "ManagerBasedRLEnv",
    env_ids: "Sequence[int]",
    command_term_name: str = "arm_pose_command",
    warmup_steps: int = 6000,
    hold_steps: int = 24000,
    pitch_amplitude: float = 1.5,
    roll_amplitude_levels: tuple = (0.0, 0.3, 0.6, 1.0),
) -> "torch.Tensor":
    """Phase 2: pitch fixed, roll amplitude ramps stepwise.

    Levels (Phase 2):
        0 (warmup): roll=0.0  — just pitch (verify Phase 1 policy works)
        1: roll=0.3  — small bilateral spread (~17°)
        2: roll=0.6  — medium spread (~34°)
        3: roll=1.0  — large spread (~57°)

    Default hold_steps=24000 (~1000 iters per level).
    Curriculum traverses in ~3250 iters from run start.

    Returns the current roll amplitude (rad) for logging.
    """
    step = env.common_step_counter

    if step < warmup_steps:
        level_idx = 0
    else:
        levels_advanced = (step - warmup_steps) // hold_steps
        level_idx = min(int(levels_advanced), len(roll_amplitude_levels) - 1)

    target_roll = float(roll_amplitude_levels[level_idx])

    command_term = env.command_manager.get_term(command_term_name)
    command_term.cfg.pitch_amplitude = pitch_amplitude   # stays constant
    command_term.cfg.roll_amplitude = target_roll

    return torch.tensor(target_roll, device=env.device)
