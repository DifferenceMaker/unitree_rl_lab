"""lm7n (2026-11-12): the encoder ladder, gate variants, on lm7l_arch_armsway @153199 (aspired server, 8192 x 7000).

lm7m_enc_ladder (local, 11-11) never promoted: survival 0.99 passed, the ABSOLUTE tracking gate (income > 0.5 x 20 = 10/s) did not
(8.1-8.5 at std .15/.10, 4-6 at .05 -- the std ladder tightens the kernel under it). Operator 2026-11-12: "enc_survive: Yes.
enc_nodecay: Worth a shot" and "what we don't want is for tracking to stop but the encoder dr continuing. That defeats the purpose."

Rows (scripts/queue/lm7n/HYPOTHESIS.md); both = lm7l_arch_armsway + gene_enc(0.003, the 13 leg + torso joints) + the ladder
(rungs 0.003 / 0.005 / 0.007 / 0.010 rad, >= 300 it per rung, none before +400):
  enc_survive  the operator's rule: promotion on SURVIVAL only (>= 90 % of the window's ended episodes timed out).
  enc_nodecay  survival + NO-DECAY: track_err's mean squared velocity error over the window's timed-out episodes must stay
               <= 1.25 x its value in the first eligible window (the parent's walk under the first rung). track_err is the L2
               price (raw value = |v_cmd - v|^2), independent of the tracking-std ladder, so the kernel tightening cannot trip it.
"""
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from unitree_rl_lab.tasks.locomotion import mdp

from .lm7l_env_cfg import lm7l_arch_armsway, gene_enc, SW_ENC01_JOINTS
from .lm7m_env_cfg import ENC_LADDER, ENC_SURVIVE, ENC_HOLD_ITERS, ENC_WARMUP_ITERS, STEPS_PER_IT, IT

NODECAY_TERM, NODECAY_TOL = "track_err", 1.25


def gene_enc_ladder_v2(cfg, nodecay: bool, levels=ENC_LADDER, joints=SW_ENC01_JOINTS, parent_it=IT):
    gene_enc(cfg, levels[0], joints)
    assert getattr(cfg.curriculum, "enc_ladder", None) is None and cfg.rewards.track_err.weight < 0, "[enc_ladder_v2] no ladder underneath; track_err = a price"
    cfg.curriculum.enc_ladder = CurrTerm(func=mdp.encoder_offset_ladder,
                                         params={"event_term_name": "encoder_offset", "levels": tuple(levels), "survive_frac": ENC_SURVIVE,
                                                 "gate_term": "track_lin_vel_xy", "gate_frac": None,
                                                 "min_hold_steps": ENC_HOLD_ITERS * STEPS_PER_IT,
                                                 "hold_until_step": (parent_it + ENC_WARMUP_ITERS) * STEPS_PER_IT,
                                                 "nodecay_term": NODECAY_TERM if nodecay else None, "nodecay_tol": NODECAY_TOL})


def lm7n_enc_survive(cfg): lm7l_arch_armsway(cfg); gene_enc_ladder_v2(cfg, nodecay=False)
def lm7n_enc_nodecay(cfg): lm7l_arch_armsway(cfg); gene_enc_ladder_v2(cfg, nodecay=True)
