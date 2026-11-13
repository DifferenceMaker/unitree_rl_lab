"""lm7p (2026-11-13): the s1_pilot settings + the encoder ladder, both directions -- the candidates for the next study's PARENT.

Operator 2026-11-13: "First train the winner of the study with the encoders AND then put it as the next study's parent" / "Or the
other way around" / "encoder ladder but i think the no decay has to stay as the safety feature" / "I wouldn't immediately hold it at
0.01" (the ladder restarts at +-0.003 in every row and climbs on survival + no-decay).

Every row = study_env_cfg.study_s1(<that trial's levers>) + lm7n.gene_enc_ladder_v2(nodecay=True): rungs +-0.003 / 0.005 / 0.007 /
0.010 rad on the 13 leg + torso joints, promotion when >= 90 % of the window's ended episodes timed out AND track_err's raw value per
second stays <= 1.25 x the first eligible window's; >= 300 it per rung, none before +400 (earliest top rung at +1176).

  A = a study trial's CHECKPOINT + the ladder (the trial's own reward settings stay; the band is the only change):
      lm7p_t21_enc  warmstart s1_pilot_t21 @160198   (track_err_w -1.51, std_floor .091, guard .94: stillest stand, best final tracking)
      lm7p_t36_enc  warmstart s1_pilot_t36 @160198   (track_err_w -0.5, std_floor .115: the most even tracking over its four exams)
      lm7p_t37_enc  warmstart s1_pilot_t37 @160198   (track_err_w -0.5, std_floor .079: the best homer)
  B = the band-trained walker + the study's settings ("the other way around"):
      lm7p_enc_best warmstart lm7n_enc_survive @160198 (arch_armsway + the band at +-0.010) + the model's best-guess settings at 37
                    results (track_w 10, track_err_w -1.4473, std_floor .0817, no guard, yaw_err_w -12). The reward change is the lever.
All four parents end at iteration 160198, so hold_until_step is the same in every row.
"""
from .lm7n_env_cfg import gene_enc_ladder_v2
from .study_env_cfg import study_s1

IT = 160198   # the last checkpoint of every s1_pilot trial and of lm7n_enc_survive
LEVERS = {
    "t21":  dict(track_w=10.0, track_err_w=-1.5077, std_floor=0.0912, guard_w=0.9392, yaw_err_w=-12.0),   # scripts/study/jobs/s1_pilot/s1_pilot_t21.job
    "t36":  dict(track_w=10.0, track_err_w=-0.5,    std_floor=0.1147, guard_w=0.0,    yaw_err_w=-12.0),   # ... _t36.job
    "t37":  dict(track_w=10.0, track_err_w=-0.5,    std_floor=0.0794, guard_w=0.0,    yaw_err_w=-12.0),   # ... _t37.job
    "best": dict(track_w=10.0, track_err_w=-1.4473, std_floor=0.0817, guard_w=0.0,    yaw_err_w=-12.0),   # GP best guess, 37 results (watch.log 2026-11-13 09:51)
}


def _row(cfg, key):
    study_s1(cfg, **LEVERS[key])                              # lm7l_arch_armsway + the five levers (enc_scale 0 = no band underneath)
    gene_enc_ladder_v2(cfg, nodecay=True, parent_it=IT)       # + the band as a ladder, survival + no-decay


def lm7p_t21_enc(cfg): _row(cfg, "t21")
def lm7p_t36_enc(cfg): _row(cfg, "t36")
def lm7p_t37_enc(cfg): _row(cfg, "t37")
def lm7p_enc_best(cfg): _row(cfg, "best")
