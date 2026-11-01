"""lm7i (2026-11-01): the three focus items on lm7h_b_h4_still -- no counter-sway after a push, arms locked, calm recoveries.

Operator 2026-10-31 sim2sim (scripts/queue/lm7h/CONCLUSIONS.md, session_2026-10-31): lm7h_a_h4_still "coutner sways when pushed like
mad"; arms "move left to right like pendulums. They need to be locked in in place"; lm7h_b_h4_still "default stance seems better ...
calmer ... Recoveries are not good. They are chaotic ... I hate the anti-swaying a lot". "Let's focus on these 3 things for now."
11-01: parent = lm7h_b_h4_still_2026-10-29 @119395; "Your proposed were good. Add mine on top of yours"; "We try both".

Genes (scripts/queue/lm7i/HYPOTHESIS.md; every gate is the COMMAND, Bible rule 32):
  arms      arm_swing removed (operator: "Let's remove this term ... The swing is lateral anyways"); joint_deviation_arms -1.0 -> -3.0
            (L1 to default, the same arm joints). Parent ledger @119390: joint_deviation_arms -1.46/s (L1 sum 1.46 rad = ~6 deg per
            joint), arm_swing +0.38/s -> at the parent's behaviour the delta is -2.9 - 0.4 = -3.3/s (a price delta: Bible rule 26 watch).
  softstop  track_lin_vel_xy -> mdp.track_lin_vel_xy_split: std_stand .50 at |cmd| < .03 (p14h's kernel width), the ladder's std
            (.15 -> .10 -> .05, the parent ended at .05) while walking; weight 20 unchanged (no income change at a standstill).
  upright6  upright_bonus (upright_bonus_standing, std .05, thr .03) 3.0 -> 6.0 (p14e_stack_up6's dose; parent @119390 +0.50/s -> ~+0.5/s).
  energy    lm7g gene_energy_gated: energy_cmd_gated -0.01 (full at a zero command, half once |cmd| >= 0.1)
  energycot lm7g gene_energy_cot: energy_per_distance -0.01 (the plain price at a stand, halved at 0.5 m/s)
Rows: arms | softstop | upright6 | soft_up6 (softstop + upright6) | energy | energycot | combo_up6 (arms + upright6 + energy) |
      combo_soft (arms + softstop + energy). Every row = lm7h_env_cfg.b_h4 (parent B + fric + bank_sw + stance_p14h + no_step +
      sway_still + tilt .8) + its genes.
"""
from unitree_rl_lab.tasks.locomotion import mdp

from .lm7g_env_cfg import gene_energy_cot, gene_energy_gated
from .lm7h_env_cfg import THR, b_h4

PARENT, IT = "lm7h_b_h4_still_2026-10-29", 119395
STD_STAND = 0.5   # p14h_stancewide's track_lin_vel_xy std


def gene_arms(cfg):
    R = cfg.rewards
    a, d = getattr(R, "arm_swing", None), R.joint_deviation_arms
    assert a is not None and a.func is mdp.arm_gait_swing and a.weight == 1.0, "[arms] the parent's arm_swing +1.0 (arm_gait_swing)"
    assert d.func is mdp.joint_deviation_l1 and d.weight == -1.0, "[arms] the parent's joint_deviation_arms -1.0 (L1)"
    R.arm_swing = None
    d.weight = -3.0


def gene_softstop(cfg):
    t = cfg.rewards.track_lin_vel_xy
    assert t.func.__name__ == "track_lin_vel_xy_yaw_frame_exp" and t.weight == 20.0, "[softstop] the parent's track_lin_vel_xy 20 (yaw-frame exp)"
    assert t.params["std"] == 0.15 and t.params["command_name"] == "base_velocity", "[softstop] the ladder's .15 start on base_velocity"
    assert cfg.curriculum.track_std.params["reward_terms"][0] == "track_lin_vel_xy", "[softstop] the ladder mutates track_lin_vel_xy.params['std']"
    t.func = mdp.track_lin_vel_xy_split
    t.params = {"std": t.params["std"], "std_stand": STD_STAND, "command_name": "base_velocity", "cmd_thr": THR}


def gene_upright6(cfg):
    u = cfg.rewards.upright_bonus
    assert u.func is mdp.upright_bonus_standing and u.weight == 3.0 and u.params["std"] == 0.05 and u.params["thr"] == THR, \
        "[upright6] the parent's upright_bonus_standing 3.0 @ std .05, thr .03"
    u.weight = 6.0


def arms(cfg): b_h4(cfg); gene_arms(cfg)
def softstop(cfg): b_h4(cfg); gene_softstop(cfg)
def upright6(cfg): b_h4(cfg); gene_upright6(cfg)
def soft_up6(cfg): b_h4(cfg); gene_softstop(cfg); gene_upright6(cfg)
def energy(cfg): b_h4(cfg); gene_energy_gated(cfg)
def energycot(cfg): b_h4(cfg); gene_energy_cot(cfg)
def combo_up6(cfg): b_h4(cfg); gene_arms(cfg); gene_upright6(cfg); gene_energy_gated(cfg)
def combo_soft(cfg): b_h4(cfg); gene_arms(cfg); gene_softstop(cfg); gene_energy_gated(cfg)
