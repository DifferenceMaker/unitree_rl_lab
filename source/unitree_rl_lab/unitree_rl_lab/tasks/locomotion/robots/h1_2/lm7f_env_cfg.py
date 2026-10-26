"""lm7f (2026-10-26): the lm7e sim2sim keepers composed on lm7e_A_kit, plus the policy-activation DR.

Parent for EVERY row: lm7e_A_kit_2026-10-22 @98398 (operator 2026-10-26: "I think the A_kit can be the parent for all of the axis
runs"; "worth going further with": small steps 0.05-0.15, sideways best by far, the honest turn at -0.5). parent_kit = parent_A +
gene_kit, so hips + footbonus + clear08 ride in every row ("make sure it keeps being there").

Keepers from the lm7e sim2sim (verbatim in scripts/queue/lm7e/CONCLUSIONS.md): gforce (greenlit: "exactly what i want" at 0.10),
angmom (torso stability 3.00), resets ("we need to add this"), posture ("there really is something here"), airtime (net positive vs
the parent). Composed as a LADDER, one gene per rung (p14c: flat combos blew up, sequential composition attributes the break).
Axes on the kit: clear9 (clearance amped), hips4 (the proactive-turn + stance gene doubled), footbonus6 / footbonus6_still ("amp
it up"), stand10 (the operator's 0.25 -> 0.10 default proposal), alive10 / alive5.
Activation DR, three rows on kit_gforce: resets (lm7e, 0.05 rad), activation (v1: wider blind offsets + base tilt and velocity),
bank (v2: 30 % of episodes start from the balance policy's recorded stance, `mdp.reset_from_pose_bank`).
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from unitree_rl_lab.tasks.locomotion import mdp

from .lm7e_env_cfg import gene_airtime, gene_angmom, gene_gforce, gene_kit, gene_posture, gene_resets, parent_A

PARENT, IT = "lm7e_A_kit_2026-10-22", 98398
BANK = "balance_p14g_hips_armfull_2026-09-23.pt"   # aspired-isaac-lab/assets/pose_banks/, recorded by scripts/tools/record_pose_bank.py


# ── parent ──
def parent_kit(cfg):
    parent_A(cfg); gene_kit(cfg)


# ── gait / stance axes ──
def gene_clear9(cfg):
    """The plain clearance 6 -> 9 at the kit's .08 (operator: "clearance seems to be the main thing for small velocity stepping ...
    we also need a run where the clearance ... was amped up"). The swing bonus (3 @ .08) is untouched."""
    t = cfg.rewards.feet_clearance
    assert t.weight == 6.0 and t.params["target_height"] == 0.08, "[clear9] from clear6 @ .08 (the kit)"
    t.weight = 9.0


def gene_hips4(cfg):
    """joint_deviation_hips L1 -2 -> -4: lm7e_A_hips turned most proactively and is a stance lever; the one rotation row of the wave."""
    t = cfg.rewards.joint_deviation_hips
    assert t.weight == -2.0, "[hips4] from the kit's -2"
    t.weight = -4.0


def gene_footbonus6(cfg):
    t = cfg.rewards.feet_yaw_bonus
    assert t.weight == 3.0, "[footbonus6] from the kit's 3"
    t.weight = 6.0


def gene_footbonus6_still(cfg):
    """feet_yaw_bonus 6 at zero command, 3 under a command (cmd_scale .5): parallel feet where they matter, the walking toe-out free."""
    t = cfg.rewards.feet_yaw_bonus
    assert t.weight == 3.0 and t.func is mdp.feet_yaw_bonus_turn_faded, "[footbonus6_still] from the kit's turn-faded bonus @ 3"
    t.func = mdp.feet_yaw_bonus_split
    t.weight = 6.0
    t.params["cmd_thr"] = 0.03
    t.params["cmd_scale"] = 0.5


def gene_stand10(cfg):
    """rel_standing_envs .25 -> .10 (operator: "decreasing our current walking from 0.25 to, i'd say, 0.1 ... gives extra 15 % of
    walking information"); the gait term is left as the kit's (stand5's stance-intent swap is NOT part of this axis)."""
    C = cfg.commands.base_velocity
    assert C.rel_standing_envs == 0.25, "[stand10] from 25 % standing envs"
    C.rel_standing_envs = 0.10


def _alive(cfg, w):
    assert cfg.rewards.alive.weight == 30.0, "[alive] from 30"
    cfg.rewards.alive.weight = w


def gene_alive10(cfg): _alive(cfg, 10.0)
def gene_alive5(cfg): _alive(cfg, 5.0)


# ── activation DR ──
def gene_activation(cfg):
    """v1, blind offsets: joints +-0.15 rad (8.6 deg, the stance difference between the two policies) / +-1 rad/s, base roll and
    pitch +-5 deg, base velocity +-0.3 m/s and +-0.5 rad/s at reset, a random heading (the world-frame push then hits every body
    side evenly), tilt termination 0.8 rad. No xy spawn spread (operator: it trains nothing, the policy has no world frame)."""
    rb = cfg.events.reset_base.params
    assert tuple(rb["pose_range"]["x"]) == (-0.1, 0.1) and tuple(rb["velocity_range"]["x"]) == (0.0, 0.0), "[activation] from the trunk's still reset"
    rb["pose_range"]["roll"] = (-0.087, 0.087)
    rb["pose_range"]["pitch"] = (-0.087, 0.087)
    rb["pose_range"]["yaw"] = (-3.14, 3.14)
    for k in ("x", "y"):
        rb["velocity_range"][k] = (-0.3, 0.3)
    for k in ("roll", "pitch", "yaw"):
        rb["velocity_range"][k] = (-0.5, 0.5)
    cfg.events.reset_robot_joints = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                                              params={"position_range": (-0.15, 0.15), "velocity_range": (-1.0, 1.0)})
    cfg.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


def gene_bank(cfg):
    """v2, the handover itself: 30 % of the resetting envs start from a sample of the balance policy's recorded stance (q, qd,
    height, roll/pitch, base velocity; + 0.03 rad / 0.3 rad/s blind noise), appended after the trunk's reset terms so it overwrites
    their draw for the picked envs. Tilt termination 0.8 rad as on the other two activation rows."""
    assert getattr(cfg.events, "reset_pose_bank", None) is None, "[bank] no bank term underneath"
    cfg.events.reset_pose_bank = EventTerm(func=mdp.reset_from_pose_bank, mode="reset",
                                           params={"bank": BANK, "prob": 0.3, "pos_noise": 0.03, "vel_noise": 0.3})
    cfg.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


# ── compositions (the ladder) ──
def gene_gforce_resets(cfg): gene_gforce(cfg); gene_resets(cfg)
def gene_gforce_resets_posture(cfg): gene_gforce_resets(cfg); gene_posture(cfg)
def gene_gforce_resets_angmom(cfg): gene_gforce_resets(cfg); gene_angmom(cfg)
def gene_gforce_resets_angmom_posture(cfg): gene_gforce_resets(cfg); gene_angmom(cfg); gene_posture(cfg)
def gene_full(cfg): gene_gforce_resets(cfg); gene_angmom(cfg); gene_posture(cfg); gene_airtime(cfg)
def gene_gforce_airtime(cfg): gene_gforce(cfg); gene_airtime(cfg)
def gene_gforce_activation(cfg): gene_gforce(cfg); gene_activation(cfg)
def gene_gforce_bank(cfg): gene_gforce(cfg); gene_bank(cfg)
