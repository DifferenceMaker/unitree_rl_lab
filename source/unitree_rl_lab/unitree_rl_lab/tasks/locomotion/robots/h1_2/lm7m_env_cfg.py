"""lm7m (2026-11-11): the ENCODER LADDER on lm7l_arch_armsway @153199 (authored + smoked; NOT enqueued -- the operator enqueues).

Operator 2026-11-11: "the robot would first have to learn how to walk with encoder values domain randomized for 0.003 or something. Then
0.005. Then 0.007. and then 0.01, which is where we are at with the balance policy. If it can survive with 0.003 for 90% of the robot runs
then it goes up a level." One run, arch_armsway (the s1_pilot centre, model_153199, v4 anchor 57.0; operator 11-09: "Its gait when walking
is precisely what i want") + the ladder; trained on the local 5080 (4096 envs).

Gene (scripts/queue/lm7m/HYPOTHESIS.md):
  enc_ladder  gene_enc (lm7l) with p16c_sw_enc01's 13 joints (every leg joint + the torso) at the FIRST rung +-0.003 rad, plus
              curriculum.enc_ladder = mdp.encoder_offset_ladder: rungs (0.003, 0.005, 0.007, 0.010) rad, promotion when >= 90 % of the
              ended episodes timed out (no fall) AND track_lin_vel_xy's mean income per second > 0.5 x its weight (the std ladder's gate),
              after >= 300 iterations on the rung (7200 env steps); no promotion before parent_it + 300 + 100 (the critic warm-up + 100).
              Reaching 0.010 takes >= 3 x 300 iterations after the hold: the earliest full band is at ~+1300 of 7000.
"""
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from unitree_rl_lab.tasks.locomotion import mdp

from .lm7l_env_cfg import lm7l_arch_armsway, gene_enc, SW_ENC01_JOINTS
from .lm7u_env_cfg import LM7U_LADDER_GATE

PARENT, IT = "lm7l_arch_armsway_2026-11-05", 153199
ENC_LADDER = (0.003, 0.005, 0.007, 0.010)   # rad: 0.17 / 0.29 / 0.40 / 0.57 deg per joint per episode, uniform; the top = p16c_sw_enc01's band
ENC_SURVIVE = 0.90                          # operator: "survive ... for 90% of the robot runs"
ENC_HOLD_ITERS = 300                        # iterations on a rung before the next check (one window of ~7 episodes per env)
ENC_WARMUP_ITERS = 300 + 100                # --critic_warmup_iters 300 + 100: the actor must walk the first rung before any promotion
STEPS_PER_IT = 24


def gene_enc_ladder(cfg, levels=ENC_LADDER, joints=SW_ENC01_JOINTS, parent_it=IT):
    gene_enc(cfg, levels[0], joints)        # the band mechanism (Enc action + encoder-frame policy obs + the reset event) at the first rung
    assert getattr(cfg.curriculum, "enc_ladder", None) is None and cfg.curriculum.track_std.params["gate_frac"] == LM7U_LADDER_GATE, "[enc_ladder] no ladder underneath; the std ladder's gate"
    cfg.curriculum.enc_ladder = CurrTerm(func=mdp.encoder_offset_ladder,
                                         params={"event_term_name": "encoder_offset", "levels": tuple(levels), "survive_frac": ENC_SURVIVE,
                                                 "gate_term": "track_lin_vel_xy", "gate_frac": LM7U_LADDER_GATE,
                                                 "min_hold_steps": ENC_HOLD_ITERS * STEPS_PER_IT,
                                                 "hold_until_step": (parent_it + ENC_WARMUP_ITERS) * STEPS_PER_IT})
    assert cfg.events.encoder_offset.params["offset_range"] == (-levels[0], levels[0]), "[enc_ladder] starts at the first rung"


def lm7m_enc_ladder(cfg): lm7l_arch_armsway(cfg); gene_enc_ladder(cfg)
