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
 
def sustained_push_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: "Sequence[int]",
    event_term_name: str = "sustained_push_apply",
    warmup_steps: int = 10000,
    hold_steps: int = 8000,
    levels: tuple = (
        ((0.0, 0.0), (0.0, 0.0)),
        ((5.0, 15.0), (1.0, 2.0)),
        ((10.0, 25.0), (1.5, 3.0)),
        ((15.0, 50.0), (2.0, 4.0)),
    ),
) -> "torch.Tensor":
    """Stepwise curriculum for sustained external force magnitude and duration.

    Holds at levels[0] (no push) for `warmup_steps` to give the policy
    time to adapt to other new rewards (foot_stance_tracking). Then
    advances one level every `hold_steps`.

    With num_steps_per_env=24:
        - 10000 warmup steps ≈ 417 iters
        - 8000 hold steps per level ≈ 333 iters
        - 4 levels = ~1750 iters to reach max sustained push

    Returns current max force magnitude (N) for logging.
    """
    step = env.common_step_counter
    if step < warmup_steps:
        force_range, duration_range = levels[0]
    else:
        levels_advanced = (step - warmup_steps) // hold_steps
        level_idx = min(int(levels_advanced), len(levels) - 1)
        force_range, duration_range = levels[level_idx]

    event_term = env.event_manager.get_term_cfg(event_term_name)
    event_term.params["force_magnitude_range"] = force_range
    event_term.params["duration_range_s"] = duration_range

    return torch.tensor(force_range[1], device=env.device)