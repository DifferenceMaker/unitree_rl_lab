"""lm7r (2026-11-20): the s4_body_care trials + the encoder ladder, both directions (the lm7p pattern on s4) -- 6 rows, 3500 it.

Operator 2026-11-20: "Send out an agent to author and smoke these bad boys" (results today -> 3500 iterations).

Every row = study_env_cfg.study_s4(<nine weights>) + lm7q_env_cfg.gene_enc_ladder_up (rungs +-0.010 -> 0.012 -> 0.015 rad on the
13 leg + torso joints, promotion = survival >= 0.90 + no-decay on track_vx x 1.25, >= 300 it per rung, none before parent_it + 400).
Neither gene is modified here: study_s4 builds the band HELD at +-0.010 (S2_BAND) underneath, which is the ladder's first rung, so
gene_enc_ladder_up's "band +-0.010 underneath" assert holds for every row; only `parent_it` changes (hold_until_step).

  A  (an s4 trial + the ladder; its nine weights exactly as its job file passed them; iteration 0 = the trial's world):
     lm7r_t27_enc     s4_body_care_t27_2026-11-17 @172425 (the harvest pick; the run ended at 172647)
     lm7r_t28_enc     s4_body_care_t28_2026-11-17 @172647
     lm7r_t35_enc     s4_body_care_t35_2026-11-17 @172647
  B  (the encoder policy lm7q_enc_up_2026-11-16 @174647 = t57 + the ladder to +-0.015, trained at 0.015 for its last 4147 it, + new
     weights; the ladder RESTARTS at +-0.010 = the lm7p enc_best pattern -- the ladder's level lives on the env object, it is not in
     the checkpoint, so every warmstart starts on rung 0 whatever band its parent ended on):
     lm7r_encup_gpbest  s4's final GP best guess (watch.log `gp: best guess`, 88 results, ~ 77.66 +- 2.89)
     lm7r_encup_t27w    t27's nine weights
     lm7r_encup_t28w    t28's nine weights

Timing (STEPS_PER_IT 24, a check only at a 60 s episode boundary = every 125 it; the first episode after the resume is skipped):
  @172425 (t27): counter 4138200 -> boundaries +75, +200, +325, +450 ... -> 0.012 at +450 at the earliest, 0.015 at +825
  @172647 (t28 / t35) and @174647 (B rows; 2000 it apart = 16 episodes): boundaries +103, +228, +353, +478 -> 0.012 at +478, 0.015
  at +853 -> ~2650 of 3500 it on the top rung.
"""
from .lm7q_env_cfg import gene_enc_ladder_up
from .study_env_cfg import study_s4, S4_PRICES

T27_MS, T27_IT = "s4_body_care_t27_2026-11-17", 172425
T28_MS, T28_IT = "s4_body_care_t28_2026-11-17", 172647
T35_MS, T35_IT = "s4_body_care_t35_2026-11-17", 172647
LM7Q_MS, LM7Q_IT = "lm7q_enc_up_2026-11-16", 174647

# the trials' nine levers, to the digit of scripts/study/jobs/s4_body_care/s4_body_care_t<N>.job's edit_raw (== the db's `levers`)
S4_T27 = dict(track_w=9.771, arm_dev_w=-8.0824, arm_motion_w=-0.7735, force_cap_w=-0.0112, action_rate_w=-0.5079, joint_acc_w=-1.689e-07,
              step_price_w=-2.0174, spawn_hold_w=-0.3564, contact_w=-2.8501)
S4_T28 = dict(track_w=10.5824, arm_dev_w=-8.5399, arm_motion_w=-0.8474, force_cap_w=-0.0162, action_rate_w=-0.647, joint_acc_w=-1.393e-07,
              step_price_w=-1.4917, spawn_hold_w=-0.6154, contact_w=-1.1249)
S4_T35 = dict(track_w=11.4651, arm_dev_w=-11.8669, arm_motion_w=-1.0, force_cap_w=-0.0342, action_rate_w=-0.5646, joint_acc_w=-7.159e-07,
              step_price_w=-0.7581, spawn_hold_w=-2.0066, contact_w=-2.7593)
# studies/s4_body_care/watch.log, the last `gp: best guess` line ([08:30:50], 88 results): ~ 77.66 +- 2.89
S4_GPBEST = dict(track_w=13.95, arm_dev_w=-7.5595, arm_motion_w=-0.705, force_cap_w=-0.0392, action_rate_w=-0.7494, joint_acc_w=-8.577e-07,
                 step_price_w=-0.7885, spawn_hold_w=-2.0048, contact_w=-2.0048)

assert all(set(w) == {"track_w", *S4_PRICES} for w in (S4_T27, S4_T28, S4_T35, S4_GPBEST)), "[lm7r] every row sets all nine s4 levers"


def lm7r_row(cfg, weights: dict, parent_it: int):
    """study_s4 at the row's nine weights (the band held at +-0.010 underneath) + the lm7q ladder timed from the row's warmstart"""
    study_s4(cfg, **weights)
    gene_enc_ladder_up(cfg, parent_it=parent_it)


# A: the trial + the ladder (reward weights = the trial's own)
def lm7r_t27_enc(cfg): lm7r_row(cfg, S4_T27, T27_IT)
def lm7r_t28_enc(cfg): lm7r_row(cfg, S4_T28, T28_IT)
def lm7r_t35_enc(cfg): lm7r_row(cfg, S4_T35, T35_IT)


# B: lm7q (the encoder policy) + new weights, the ladder restarted at +-0.010
def lm7r_encup_gpbest(cfg): lm7r_row(cfg, S4_GPBEST, LM7Q_IT)
def lm7r_encup_t27w(cfg): lm7r_row(cfg, S4_T27, LM7Q_IT)
def lm7r_encup_t28w(cfg): lm7r_row(cfg, S4_T28, LM7Q_IT)
