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
    gate_frac: float | None = 0.5,
    min_hold_steps: int = 7200,
    hold_until_step: int = 0,
    nodecay_term: str | None = None,
    nodecay_tol: float = 1.25,
) -> torch.Tensor:
    """Encoder-offset band ladder: the reset event's `offset_range` = +-levels[k]; k climbs one level (never back) at an episode
    boundary (common_step_counter % max_episode_length == 0) when the rung has been held min_hold_steps env steps (24 per PPO
    iteration), the step is >= hold_until_step (the critic warm-up: no promotion under a frozen actor) and, over the episodes that ended
    since the last eligible check:
      (a) SURVIVAL: >= survive_frac of them ended by TIME-OUT (not a fall / height termination)                    -- always;
      (b) ABSOLUTE tracking gate (gate_frac not None, lm7m_enc_ladder): gate_term's mean income/s > gate_frac x its weight.
          2026-11-12: this blocked lm7m for 7000 it (income 8.1-8.5 < 10; the std ladder tightens the kernel under it) -> None in
          lm7n (operator's rule = survival only);
      (c) NO-DECAY gate (nodecay_term not None, lm7n_enc_nodecay; operator: "what we don't want is for tracking to stop but the
          encoder dr continuing"): q = nodecay_term's mean RAW value per second over the window's TIMED-OUT episodes (episode sum /
          (T x weight); for track_err = the mean squared velocity error, independent of the tracking-std ladder) must stay
          <= nodecay_tol x the BASELINE q = the first eligible window's q (the parent's walk under the first rung). Lower = better.

    Called from CurriculumManager.compute inside _reset_idx(env_ids), BEFORE termination_manager.reset: termination_manager.time_outs
    still holds this step's flags for env_ids. Episodes ending in the first max_episode_length steps of the run are skipped (their
    sums are partial: the random initial episode length). The resume path calls compute(env_ids=None) -> a slice: skipped.
    Returns the current half-width (rad) for the Curriculum/ log.
    """
    if not hasattr(env, "_enc_ladder"):
        env._enc_ladder = {"level": 0, "n": 0, "n_timeout": 0, "income": 0.0, "q": 0.0, "q_base": None,
                           "hold_start": env.common_step_counter, "first_step": env.common_step_counter}
        env.event_manager.get_term_cfg(event_term_name).params["offset_range"] = (-float(levels[0]), float(levels[0]))
    S = env._enc_ladder
    step = env.common_step_counter
    if isinstance(env_ids, slice) or len(env_ids) == 0:
        S["first_step"] = S["hold_start"] = step                     # (compute(None) on resume: the run's real first step)
        return torch.tensor(levels[S["level"]], device=env.device)
    if step - S["first_step"] <= env.max_episode_length:
        return torch.tensor(levels[S["level"]], device=env.device)

    # accumulate the ended episodes: how they ended + what the gate terms paid them
    rm = env.reward_manager
    to = env.termination_manager.time_outs[env_ids]
    S["n"] += int(len(env_ids))
    S["n_timeout"] += int(to.sum().item())
    S["income"] += float(rm._episode_sums[gate_term][env_ids].sum().item()) / env.max_episode_length_s
    if nodecay_term is not None:
        w = rm.get_term_cfg(nodecay_term).weight
        S["q"] += float(rm._episode_sums[nodecay_term][env_ids][to].sum().item()) / (env.max_episode_length_s * w)

    at_boundary = step % env.max_episode_length == 0
    if at_boundary and step >= hold_until_step and step - S["hold_start"] >= min_hold_steps and S["n"] > 0:
        survive = S["n_timeout"] / S["n"]
        income = S["income"] / S["n"]
        q = S["q"] / max(1, S["n_timeout"])
        if nodecay_term is not None and S["q_base"] is None:
            S["q_base"] = q
            print(f"[enc_ladder] step {step}: no-decay baseline {nodecay_term} q = {q:.5f} (limit {nodecay_tol} x = {nodecay_tol * q:.5f})")
        ok = survive >= survive_frac
        why = f"survive {survive:.3f} (>= {survive_frac})"
        if gate_frac is not None:
            gate = rm.get_term_cfg(gate_term).weight * gate_frac
            ok = ok and income > gate
            why += f", income {income:.3f} (> {gate:.3f})"
        if nodecay_term is not None:
            ok = ok and q <= nodecay_tol * S["q_base"]
            why += f", {nodecay_term} q {q:.5f} (<= {nodecay_tol * S['q_base']:.5f})"
        if S["level"] < len(levels) - 1:
            if ok:
                S["level"] += 1
                S["hold_start"] = step
                half = float(levels[S["level"]])
                env.event_manager.get_term_cfg(event_term_name).params["offset_range"] = (-half, half)
                print(f"[enc_ladder] step {step}: PROMOTED to level {S['level']} = +-{half:.3f} rad: {why} over {S['n']} episodes")
            else:
                print(f"[enc_ladder] step {step}: HELD at +-{float(levels[S['level']]):.3f} rad: {why} over {S['n']} episodes")
        S["n"] = 0; S["n_timeout"] = 0; S["income"] = 0.0; S["q"] = 0.0      # the window = the episodes since the last eligible check
    return torch.tensor(levels[S["level"]], device=env.device)
