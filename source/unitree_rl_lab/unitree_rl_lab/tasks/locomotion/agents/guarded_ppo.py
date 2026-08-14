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
"""

from __future__ import annotations

import torch

from rsl_rl.algorithms.ppo import PPO


class GuardedPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nonfinite_skips_total = 0
        self._nonfinite_skips_window = 0

        orig_step = self.optimizer.step

        def guarded_step(*sargs, **skwargs):
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        self._nonfinite_skips_total += 1
                        self._nonfinite_skips_window += 1
                        self.optimizer.zero_grad()
                        return None
            return orig_step(*sargs, **skwargs)

        self.optimizer.step = guarded_step

    def update(self) -> dict[str, float]:
        self._nonfinite_skips_window = 0
        loss_dict = super().update()
        loss_dict["nonfinite_skips"] = float(self._nonfinite_skips_window)
        return loss_dict
