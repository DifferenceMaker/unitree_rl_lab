"""lm7m (2026-11-11): the ENCODER-OFFSET LADDER -- the p16c_sw_enc01 band introduced as a curriculum instead of a fixed dose.

Operator 2026-11-11: "What if we increased the encoder dr sequentially? Like, the robot would first have to learn how to walk with encoder
values domain randomized for 0.003 or something. Then 0.005. Then 0.007. and then 0.01, which is where we are at with the balance policy.
If it can survive with 0.003 for 90% of the robot runs then it goes up a level."

Why a ladder: a fixed +-0.03 band broke training (p16_dr enc03, memory encoder-dr-destabilizer), the fixed +-0.01 band fixed the balance
policy on the robot (p16c_sw_enc01) but the walk rows carrying it (lm7l enc01 x3) walked badly. A warmstart meets the band as an income
delta on day one; the ladder lets the policy earn its way up, the same mechanism as the tracking-std ladder (mdp.lm7u.tracking_std_levels):
performance-gated, never back, state on the env, so --resume_curriculum starts it at levels[0].

Mechanism untouched: mdp.dr.randomize_encoder_offset draws the per-joint constant offset at reset, uniform in the event's `offset_range`;
the ladder only rewrites that range. The policy observes q + offset, the target lands at target - offset, the critic sees the true angles.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def encoder_offset_ladder(
    env: "ManagerBasedRLEnv",
    env_ids,
    event_term_name: str = "encoder_offset",
    levels: tuple = (0.003, 0.005, 0.007, 0.010),
    survive_frac: float = 0.90,
    gate_term: str = "track_lin_vel_xy",
    gate_frac: float = 0.5,
    min_hold_steps: int = 7200,
    hold_until_step: int = 0,
) -> torch.Tensor:
    """Encoder-offset band ladder: the reset event's `offset_range` = +-levels[k]; k climbs one level (never back) when, over the
    episodes that ended since the last check, (a) at least survive_frac of them ended by TIME-OUT (not a fall / height termination)
    and (b) the gate term's mean episode income per second exceeds gate_frac x its weight (the tracking-std ladder's own gate), and the
    current level has been held for min_hold_steps env steps (24 per PPO iteration). Checks happen at episode boundaries
    (common_step_counter % max_episode_length == 0, where every env resets under the fixed-length episodes), never before
    hold_until_step (the critic warm-up: a promotion under a frozen actor would hand the unfrozen actor a band it never walked in).

    Called from CurriculumManager.compute inside _reset_idx(env_ids), BEFORE termination_manager.reset: termination_manager.time_outs
    still holds this step's flags for env_ids. The resume path calls compute(env_ids=None) -> a slice: no episode ended, skipped.
    Returns the current half-width (rad) for the Curriculum/ log.
    """
    if not hasattr(env, "_enc_ladder"):
        env._enc_ladder = {"level": 0, "n": 0, "n_timeout": 0, "income": 0.0, "hold_start": env.common_step_counter}
        env.event_manager.get_term_cfg(event_term_name).params["offset_range"] = (-float(levels[0]), float(levels[0]))
    S = env._enc_ladder
    if isinstance(env_ids, slice) or len(env_ids) == 0:
        return torch.tensor(levels[S["level"]], device=env.device)

    # accumulate the ended episodes: how they ended + what the gate term paid them
    S["n"] += int(len(env_ids))
    S["n_timeout"] += int(env.termination_manager.time_outs[env_ids].sum().item())
    S["income"] += float(env.reward_manager._episode_sums[gate_term][env_ids].sum().item()) / env.max_episode_length_s

    step = env.common_step_counter
    at_boundary = step % env.max_episode_length == 0
    if at_boundary and S["level"] < len(levels) - 1 and step >= hold_until_step and step - S["hold_start"] >= min_hold_steps and S["n"] > 0:
        survive = S["n_timeout"] / S["n"]
        income = S["income"] / S["n"]
        gate = env.reward_manager.get_term_cfg(gate_term).weight * gate_frac
        if survive >= survive_frac and income > gate:
            S["level"] += 1
            S["hold_start"] = step
            half = float(levels[S["level"]])
            env.event_manager.get_term_cfg(event_term_name).params["offset_range"] = (-half, half)
            print(f"[enc_ladder] step {step}: level {S['level']} -> +-{half:.3f} rad (survive {survive:.3f} >= {survive_frac}, "
                  f"income {income:.3f} > {gate:.3f} over {S['n']} episodes)")
        S["n"] = 0; S["n_timeout"] = 0; S["income"] = 0.0      # the window = the episodes since the last check
    return torch.tensor(levels[S["level"]], device=env.device)
