"""lm7o (2026-11-12): the encoder ladder on lm7k_micro_legs_squeeze @146200 -- the walker the robot has shown working.

Operator 2026-11-12, after lm7m_enc_ladder (arch_armsway + a fixed +-0.003 band) on the robot: "It didn't have the perpetual right
lean. But the walk wasn't stable." / "maybe we should've added the encoder dr on the micro_legs_squeeze instead of the next child?"
/ "nodecay or just survive? Both? Worth testing both imo."

Why this parent: arch_armsway (the lm7m / lm7n parent) adds the swing arch + arm sway to squeeze and loses tracking before any band
(Exam v4: squeeze 59.8, tracking 10.6 / 20; arch_armsway 57.0, tracking 7.5). On squeeze the band is the ONLY difference from a
walk that has stood and walked on the robot (11-05 / 11-11 / 11-12 handovers).

Rows (aspired-isaac-lab scripts/queue/lm7o/HYPOTHESIS.md), both = lm7l.parent (= lm7k_micro_legs_squeeze, bit-identical)
+ lm7n.gene_enc_ladder_v2 with the squeeze checkpoint's iteration:
  sq_survive  promotion on SURVIVAL only (>= 90 % of the window's ended episodes timed out) -- lm7n_enc_survive's gate.
  sq_nodecay  survival + NO-DECAY: track_err's raw value per second over the window's timed-out episodes
              (|v_xy cmd - v_xy|^2 + (wz cmd - wz)^2, NOT the linear error alone) <= 1.25 x the first eligible window's.
              On arch_armsway the guard never bound (q 0.51-0.53 vs the limit 0.66), so the two lm7n rows ran identically;
              here it binds only if the band costs squeeze's tracking more than that.
"""
from .lm7l_env_cfg import parent as squeeze, PARENT, IT
from .lm7n_env_cfg import gene_enc_ladder_v2

assert PARENT == "lm7k_micro_legs_squeeze_2026-11-05" and IT == 146200, "[lm7o] lm7l's parent = the squeeze milestone @146200"


def lm7o_sq_survive(cfg): squeeze(cfg); gene_enc_ladder_v2(cfg, nodecay=False, parent_it=IT)
def lm7o_sq_nodecay(cfg): squeeze(cfg); gene_enc_ladder_v2(cfg, nodecay=True, parent_it=IT)
