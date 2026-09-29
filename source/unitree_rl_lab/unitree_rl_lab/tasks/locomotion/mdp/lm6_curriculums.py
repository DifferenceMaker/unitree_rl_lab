"""LM6B (2026-09-29): push curricula gated on COMPETENCE, not on the clock.

The step-clocked push_velocity_curriculum / sustained_push_curriculum advance one level every
hold_steps (3000 env steps = ~125 PPO iterations on the balance trunk), so a SCRATCH walker met
1.5 m/s pushes by iteration ~1000, when it had just learned to stand: every lm6 episode ended at
~10.6 s = the first push (episode length ~530 steps, base_height termination 1.00, time_out 0.00,
action std frozen at 0.96 for 9000 iterations, lin_vel_levels never left 0.40). A warmstarted
walker inherited push competence from its balance parent and never showed this; from scratch the
clock is the wrong gate.

Gate = COMPETENCE, the way lin_vel_cmd_levels gates the command ranges (upstream unitree h1/g1
style). Two metrics, both exponentially weighted over the recent ~`min_resets` episode ends:
  * tracking: mean(episode_sums[gate_reward_term][env_ids]) / max_episode_length_s / weight -- the
    upstream metric, which couples tracking QUALITY with SURVIVAL (a perfect tracker dying at 40 % of
    the episode reads 0.4); must reach `gate_frac` ("the pushing comes after the tracking is fine",
    operator 2026-09-29);
  * survival: the fraction of episode ends that were TIME-OUTS; must reach `survival_frac`.
A level advances only when both pass, at least `min_resets` episodes have ended and at least
`min_hold_steps` env steps have passed since the last change, after `warmup_steps`. Levels never
descend. `gate_reward_term=None` = survival only. Returns the current level for logging under the
same Curriculum/<name> key as before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _competence_gate(env, env_ids, key: str, n_levels: int, gate_reward_term, gate_frac: float,
                     survival_term: str, survival_frac: float, min_resets: int, min_hold_steps: int,
                     warmup_steps: int) -> int:
    """Shared state machine: returns the current level index after accounting this reset batch."""
    states = getattr(env, "_lm6_gate_states", None)
    if states is None:
        states = env._lm6_gate_states = {}
    st = states.get(key)
    step = int(env.common_step_counter)
    if st is None:
        st = states[key] = {"level": 0, "last_change": step, "resets": 0, "surv": 0.0, "trk": 0.0}
    n = int(len(env_ids))
    if n > 0:
        # exponentially-weighted metrics with a memory of ~min_resets episode ends (NOT cumulative
        # since the last change: a long dying phase would otherwise bury the first survivals)
        decay = (1.0 - 1.0 / max(min_resets, 1)) ** n
        to = env.termination_manager.get_term(survival_term)[env_ids].float().mean().item()
        st["surv"] = st["surv"] * decay + to * (1.0 - decay)
        if gate_reward_term is not None:
            w = float(env.reward_manager.get_term_cfg(gate_reward_term).weight)
            frac = (torch.mean(env.reward_manager._episode_sums[gate_reward_term][env_ids]).item()
                    / env.max_episode_length_s / w) if w > 0 else 0.0
            st["trk"] = st["trk"] * decay + frac * (1.0 - decay)
        st["resets"] += n
    trk_ok = True if gate_reward_term is None else st["trk"] >= gate_frac
    if (st["level"] < n_levels - 1 and step >= warmup_steps and step - st["last_change"] >= min_hold_steps
            and st["resets"] >= min_resets and st["surv"] >= survival_frac and trk_ok):
        st["level"] += 1
        st["last_change"] = step
        st["resets"] = 0
        st["surv"] = 0.0          # the new level must prove itself on fresh evidence
        st["trk"] = 0.0
    return st["level"]


def push_velocity_curriculum_gated(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    event_term_name: str = "push_robot",
    levels: tuple = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
    gate_reward_term: str | None = "track_lin_vel_xy",
    gate_frac: float = 0.5,
    survival_term: str = "time_out",
    survival_frac: float = 0.5,
    min_resets: int = 4096,
    min_hold_steps: int = 24000,
    warmup_steps: int = 24000,
) -> torch.Tensor:
    """push_velocity_curriculum with the competence gate (see module docstring). Same levels/log key."""
    lvl = _competence_gate(env, env_ids, f"{event_term_name}:vel", len(levels), gate_reward_term, gate_frac,
                           survival_term, survival_frac, min_resets, min_hold_steps, warmup_steps)
    target_vel = float(levels[lvl])
    event_term = env.event_manager.get_term_cfg(event_term_name)
    event_term.params["velocity_range"] = {"x": (-target_vel, target_vel), "y": (-target_vel, target_vel)}
    return torch.tensor(target_vel, device=env.device)


def sustained_push_curriculum_gated(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    event_term_name: str = "sustained_push_apply",
    levels: tuple = (
        ((0.0, 0.0), (0.0, 0.0)),
        ((0.0, 15.0), (1.5, 3.0)),
        ((0.0, 30.0), (2.0, 3.5)),
        ((0.0, 40.0), (2.0, 3.5)),
        ((0.0, 50.0), (2.0, 4.0)),
    ),
    gate_reward_term: str | None = "track_lin_vel_xy",
    gate_frac: float = 0.5,
    survival_term: str = "time_out",
    survival_frac: float = 0.5,
    min_resets: int = 4096,
    min_hold_steps: int = 24000,
    warmup_steps: int = 24000,
) -> torch.Tensor:
    """sustained_push_curriculum with the competence gate. Returns the current max force (N) for logging."""
    lvl = _competence_gate(env, env_ids, f"{event_term_name}:force", len(levels), gate_reward_term, gate_frac,
                           survival_term, survival_frac, min_resets, min_hold_steps, warmup_steps)
    force_range, duration_range = levels[lvl]
    event_term = env.event_manager.get_term_cfg(event_term_name)
    event_term.params["force_magnitude_range"] = tuple(force_range)
    event_term.params["duration_range_s"] = tuple(duration_range)
    return torch.tensor(float(force_range[1]), device=env.device)
