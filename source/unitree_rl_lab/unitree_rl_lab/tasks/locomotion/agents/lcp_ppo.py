"""LCP: Lipschitz-Constrained Policy (paper #14). CHAINED to GuardedPPO
2026-08-17: lm4b_lcp died of the exact value-explosion the optimizer guard
breaks (12.9 -> 2.7e14 -> NaN std), unguarded. Now every LCP run inherits
the nonfinite-grad skip.

Original notes (lm4_lcp run,
"Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies").

The paper's claim: replace output-space smoothing REWARDS (action_rate) with a
gradient penalty ``lcp_coef * E[ ||d mu / d obs||^2 ]`` — smoothness becomes a
property of the NETWORK (bounded local Lipschitz constant) instead of a bribe
in the reward channel. The lm4_lcp trunk deletes action_rate; this loss is
its replacement.

DELIBERATE DEVIATION from the paper: the penalty is optimized in ALTERNATING
steps after PPO's clipped update (K extra minibatch steps per iteration on
rollout observations), not folded into the per-minibatch PPO loss. Rationale:
folding it in requires copying rsl_rl's 200-line update() verbatim and pinning
this repo to its exact version (local and H200 must agree); the alternating
scheme touches only stable API surface (storage.observations, actor forward,
the shared optimizer). Same fixed-point argument as lazy R1 regularization in
GANs: for small coef the two schedules converge to the same constraint. The
penalty magnitude is logged (``lcp_grad_penalty``) so wandb shows whether the
constraint is active and how hard it fights the surrogate.

Gradient form matches the paper's released code: grad of mu.sum() w.r.t. obs
(row-sum of the Jacobian), squared, summed over obs dims, meaned over batch.

Referenced from the runner cfg via dotted class_name
("unitree_rl_lab.tasks.locomotion.agents.lcp_ppo:LCPPPO") — rsl_rl's
resolve_callable imports it; no monkey-patching.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO  # noqa: F401  (base of the guard)

from .guarded_ppo import GuardedPPO


class LCPPPO(GuardedPPO):
    def __init__(self, *args, lcp_coef: float = 0.05, lcp_num_steps: int = 4,
                 lcp_batch_size: int = 4096, **kwargs):
        super().__init__(*args, **kwargs)
        self.lcp_coef = lcp_coef
        self.lcp_num_steps = lcp_num_steps
        self.lcp_batch_size = lcp_batch_size

    def update(self) -> dict[str, float]:
        loss_dict = super().update()

        # rollout observations: TensorDict [num_steps, num_envs, ...] -> flat.
        # The buffer is preallocated and still holds this iteration's rollout
        # after update(); clear() only resets the write cursor.
        obs = self.storage.observations
        flat = obs.reshape(-1)
        total = flat.batch_size[0]

        last_gp = 0.0
        for _ in range(self.lcp_num_steps):
            idx = torch.randint(0, total, (min(self.lcp_batch_size, total),), device=self.device)
            mb = flat[idx].to_tensordict().detach().clone()
            leaves = {k: mb[k].requires_grad_(True) for k in mb.keys()}

            mu = self.actor(mb)  # deterministic mean actions
            grads = torch.autograd.grad(
                mu.sum(), list(leaves.values()), create_graph=True, allow_unused=True
            )
            gp = sum(
                g.square().sum(dim=-1).mean() for g in grads if g is not None
            )

            loss = self.lcp_coef * gp
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            last_gp = gp.item()

        loss_dict["lcp_grad_penalty"] = last_gp
        return loss_dict
