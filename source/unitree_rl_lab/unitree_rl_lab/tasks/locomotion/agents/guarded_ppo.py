"""GuardedPPO — skip-update-on-nonfinite-gradients (dp5 autopsy guard #3).

The dp5 NaN path (2026-08-11 autopsy, re-confirmed by dp5c_anchor 2026-08-14):
one out-of-bounds input -> value target explodes -> loss goes inf ->
``max_grad_norm`` CONVERTS inf into NaN (clip_coef = 1/inf = 0 and inf*0 =
NaN) -> weights are NaN one optimizer step later -> ``normal expects all
elements of std >= 0.0`` and the run is dead. ``torch.clamp`` never sanitizes
NaN, so no reward-side guard can catch this — the last line of defense has to
sit in front of the optimizer.

Implementation: wrap ``optimizer.step`` at construction instead of copying
rsl_rl's 200-line ``update()`` (the LCPPPO version-robustness argument). Every
step first checks all parameter gradients for finiteness; a non-finite batch
is DROPPED (grads zeroed, step skipped, counter bumped) and training continues
with the next minibatch. The skip count is appended to the loss dict as
``nonfinite_skips`` — a healthy run logs 0 forever; a nonzero value is the
tell that an input-path guard upstream (obs clip, termination) let something
through.

Referenced via dotted class_name
("unitree_rl_lab.tasks.locomotion.agents.guarded_ppo:GuardedPPO").

CRITIC WARM-UP (p14e, 2026-09-21). ``critic_warmup_updates`` (set by train.py
``--critic_warmup_iters``, default 0 = off) drops the ACTOR's gradients -- policy
net and the std parameter -- before every optimizer step for the first N calls of
``update()``, so only the critic learns. Why: every warmstart on the balance line
detonates critic-first. The loaded critic was fitted to the PARENT's reward ledger
and curriculum; the job changes both (a new upright weight, curriculum back to
warmup), so its value targets are wrong from iteration +0. Value loss is spiky
(>50) from +20 in every warmstart log, while a scratch run shows one spike in
20000 iterations; the actor then trains on advantages computed against a critic
that is wrong, its std drifts up, and the run goes. Letting the critic re-fit the
new ledger on the parent's own (frozen) behaviour first removes the mismatch the
actor would otherwise be trained against. Adam skips parameters whose ``.grad``
is None, so zeroing is done by setting the grad to None, which also keeps the
actor's Adam moments untouched for when it unfreezes. ``max_grad_norm`` clipping
in rsl_rl happens BEFORE the step and still sees the actor grads -- that only
scales the critic step slightly during warm-up and is accepted. Logged as
``Loss/critic_only`` (1/0) so the phase boundary is visible in the run.
"""

from __future__ import annotations

import torch

from rsl_rl.algorithms.ppo import PPO


class GuardedPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nonfinite_skips_total = 0
        self._nonfinite_skips_window = 0
        self.critic_warmup_updates = 0     # train.py --critic_warmup_iters; 0 = off
        self._updates_done = 0

        orig_step = self.optimizer.step

        def guarded_step(*sargs, **skwargs):
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        self._nonfinite_skips_total += 1
                        self._nonfinite_skips_window += 1
                        self.optimizer.zero_grad()
                        return None
            if self._updates_done < self.critic_warmup_updates:
                for p in self.actor.parameters():   # policy net AND std parameter
                    p.grad = None                    # Adam skips grad-None params: actor frozen
            return orig_step(*sargs, **skwargs)

        self.optimizer.step = guarded_step

    def update(self) -> dict[str, float]:
        self._nonfinite_skips_window = 0
        critic_only = self._updates_done < self.critic_warmup_updates
        loss_dict = super().update()
        loss_dict["nonfinite_skips"] = float(self._nonfinite_skips_window)
        loss_dict["critic_only"] = 1.0 if critic_only else 0.0
        self._updates_done += 1
        if critic_only and self._updates_done == self.critic_warmup_updates:
            print(f"[GuardedPPO] critic warm-up complete after {self._updates_done} iterations -- actor unfrozen",
                  flush=True)
        return loss_dict
