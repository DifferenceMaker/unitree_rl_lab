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


def reinit_critic(alg) -> dict:
    """--reinit_critic (p14e_critic_reinit, 2026-09-21): DISCARD the parent's value function.

    The seven p14e rows all keep the warmstart parent's critic and ask how to make it cope
    (warm-up, curriculum resume, no value clip, calmer actor). This is the opposite
    hypothesis: the parent critic is not stale but ANCHORED -- 62k iterations of sharp
    features and Adam moments fitted to another ledger -- and re-fitting FROM it is worse
    than fitting from zero. The scratch critic, which never saw the old ledger, is smooth
    for 20000 iterations (one spike, at init). Paired with the same 300-iteration critic
    warm-up as p14e_critic, the only difference between the two rows is keep vs discard.

    What it does: every nn.Linear in the critic gets reset_parameters() (that IS the value
    function; rsl_rl's critic is a small MLP), and the critic parameters' Adam state entries
    are deleted so the parent's second-moment scale cannot steer a random network. The
    critic's observation normaliser (if any) is left alone -- its running statistics describe
    the INPUTS and are still valid. The actor and its optimizer state are untouched.
    Value loss at iteration 1 will be LARGE by design (values start near 0 against returns
    of ~1300); that is the empirical signature that the discard happened.
    """
    critic = getattr(alg, "critic", None)
    if critic is None:
        raise SystemExit("[ERROR]: --reinit_critic: alg has no .critic (rsl_rl >= 5 expected)")
    before = {n: p.detach().clone() for n, p in critic.named_parameters()}
    n_lin = 0
    for m in critic.modules():
        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
            n_lin += 1
    changed = sum(int(not torch.equal(before[n], p.detach())) for n, p in critic.named_parameters())
    crit_ids = {id(p) for p in critic.parameters()}
    wiped = 0
    for p in list(alg.optimizer.state.keys()):
        if id(p) in crit_ids:
            del alg.optimizer.state[p]
            wiped += 1
    if changed == 0:
        raise SystemExit("[ERROR]: --reinit_critic changed nothing -- no nn.Linear found in the critic?")
    info = {"linear_layers": n_lin, "tensors_changed": changed, "tensors_total": len(before), "adam_entries_wiped": wiped}
    print(f"[INFO]: reinit_critic: {n_lin} Linear layers re-initialised, {changed}/{len(before)} critic parameter "
          f"tensors changed, {wiped} Adam state entries wiped; actor untouched. Expect a LARGE value loss at "
          f"iteration 1 (fresh critic) -- that is the discard showing.", flush=True)
    return info
