"""lm7g (2026-10-28): the lm7f keepers combined on lm7f_kit_gforce_resets, plus the stepping turn, energy on the ladder, DR + pc.

Parent for every row but the control: lm7f_kit_gforce_resets_2026-10-26 @105397 (operator 2026-10-28: "Default stance is sturdier
and better ... The torso is sturdier ... The walking is good and symmetrical and simple i like it. better than previous one";
"The parent is kit_gforce_resets @105397 ... Yes."). parent_gr = parent_kit (lm7e_A + hips + footbonus + clear08) + gforce + resets,
re-applied so every row differs from the parent milestone's env.yaml by exactly its genes (lineage_check_lm7g.py).

TRUNK = parent + airtime + clear9 (operator: "I want to see gforce + airtime + clear9 on top of kit. Maybe also
kit_gforce_resets_airtime_clear9 too?" -- airtime and gforce_airtime GREENLIT, clear9 "worth it"). Every other row = TRUNK + one gene
(Bible rule 36: compose sequentially), except trunk_still_* which stack the 10-27 rulings (alive 10, standing share) on the stance form.

Genes on the trunk:
  still       feet_yaw_bonus_split 6 / 3 (lm7f_kit_footbonus6_still, GREENLIT 10-27: the stance answer; footbonus6 always-on drew the
              feet inward)
  fric        friction DR U[.8, 1.2] alone (operator: "Let's keep fric. It helps with recoveries")
  drall       the full lm7e dr_all (the slide isolation: drall "Legs slide iwnard. Why?" -- if fric does not slide, the qd noise or
              the CoM draw is the cause)
  pc          privileged feet critic with the soft landing (critic warm-up 1000 + the ladder held for those 1000 it); warmstart =
              the padded transplant lm7f_kit_gforce_resets_pc10pad_2026-10-28 (pad_critic.py, critic 120 -> 130)
  fric_pc     fric + pc (operator: "I want a run which combines dr and pc on top of our lm7g_trunk")
  bank        30 % of resets from the balance policy's recorded stance (lm7f bank, "I think bank is fine. We can keep it")
  pivot       feet_pivot -2.0: the yaw rate of a loaded foot priced like slide (mdp/lm7g.py) -- the stepping turn
  energy_gated / energy_cot   energy ON THE LADDER: the parent sat at track_std .15 for all 7000 it (the gate_frac .5 of track_lin
              was never reached), so this is the one way energy has not been tried (lm7b_u CONCLUSIONS 3: "energy belongs ON the
              ladder, not on the fixed kernel"); forms = the operator's two from lm7b_u (full price at a zero command, halved under
              a command / cost of transport on the commanded speed), both at the Bible's -0.01 (rule 38)
  alive10 / stand15   the 10-27 rulings on trunk_still ("I'd go conservative and go with 10"; standing share .25 -> .10-.15, paired
              with a stance lever -- stand10 alone drew the legs inward, so .15 is the notch)
Control: lm7g_airtime_clear9 from lm7f_kit_gforce_airtime_2026-10-27 @105397 (+ clear9): the trunk without the resets gene.
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.locomotion import mdp

from .lm7e_env_cfg import _FEET_SENSOR, gene_airtime, gene_dr_all, gene_dr_fric, gene_gforce, gene_pc_contacts, gene_resets
from .lm7f_env_cfg import gene_alive10, gene_bank, gene_clear9, gene_footbonus6_still, parent_kit
from .lm7u_env_cfg import _FEET

PARENT, IT = "lm7f_kit_gforce_resets_2026-10-26", 105397
PARENT_CTL = "lm7f_kit_gforce_airtime_2026-10-27"           # same IT
PC10PAD = "lm7f_kit_gforce_resets_pc10pad_2026-10-28"       # PARENT with the critic's first layer padded 120 -> 130 (lm7e/pad_critic.py)


# ── parents ──
def parent_gr(cfg):
    parent_kit(cfg); gene_gforce(cfg); gene_resets(cfg)


def parent_ga(cfg):
    parent_kit(cfg); gene_gforce(cfg); gene_airtime(cfg)


# ── the trunk ──
def gene_trunk(cfg):
    gene_airtime(cfg); gene_clear9(cfg)


# ── genes on the trunk ──
def gene_still(cfg): gene_footbonus6_still(cfg)
def gene_fric(cfg): gene_dr_fric(cfg)
def gene_drall(cfg): gene_dr_all(cfg)


def gene_pc(cfg):
    """pc_contacts + the soft landing on THIS parent: hold the ladder at its level for the 1000 warm-up iterations (TRAIN_ARGS
    --critic_warmup_iters 1000). On lm7f the soft landing did not prevent the critic block at the unfreeze (99399-99706, value loss
    to 1e14) but the row recovered to a clean final window and walks ("Looks.. fine?"); same recipe, the padded warmstart is the parent."""
    gene_pc_contacts(cfg)
    t = cfg.curriculum.track_std
    assert t.params.get("hold_until_step", 0) == 0, "[pc] no hold underneath"
    t.params["hold_until_step"] = (IT + 1000) * 24


def gene_fric_pc(cfg): gene_fric(cfg); gene_pc(cfg)


def gene_pivot(cfg):
    """feet_pivot -2.0 (= feet_slide's weight): |yaw rate| of a loaded foot. Pre-flight: a constraint (a stepping turn and a straight
    walk pay ~0), gate = feet_slide's contact flag, no kernel, income delta on the warmstart = a new price on the sliding turn the
    parent does today (up to ~1.2/s at a 0.3 rad/s both-feet slide) against the yaw income it keeps either way."""
    assert getattr(cfg.rewards, "feet_pivot", None) is None and cfg.rewards.feet_slide.weight == -2.0, "[pivot] no pivot term underneath; slide at -2"
    cfg.rewards.feet_pivot = RewTerm(func=mdp.feet_pivot, weight=-2.0, params={"sensor_cfg": _FEET_SENSOR, "asset_cfg": _FEET})


def gene_energy_gated(cfg):
    """energy_cmd_gated -0.01: the full power price at a zero command, half of it once |cmd| >= 0.1 (vx, vy, wz norm)."""
    assert getattr(cfg.rewards, "energy", None) is None, "[energy] no energy term underneath"
    cfg.rewards.energy = RewTerm(func=mdp.energy_cmd_gated, weight=-0.01,
                                 params={"command_name": "base_velocity", "moving_scale": 0.5, "cmd_ref": 0.1, "asset_cfg": SceneEntityCfg("robot")})


def gene_energy_cot(cfg):
    """energy_per_distance -0.01: power x 0.25 / max(|v cmd| + 0.5 |wz cmd|, 0.25) -- the plain price at a stand, halved at 0.5 m/s."""
    assert getattr(cfg.rewards, "energy", None) is None, "[energy] no energy term underneath"
    cfg.rewards.energy = RewTerm(func=mdp.energy_per_distance, weight=-0.01,
                                 params={"command_name": "base_velocity", "v_floor": 0.25, "wz_lever": 0.5, "asset_cfg": SceneEntityCfg("robot")})


def gene_stand15(cfg):
    C = cfg.commands.base_velocity
    assert C.rel_standing_envs == 0.25, "[stand15] from 25 % standing envs"
    C.rel_standing_envs = 0.15


# ── compositions ──
def gene_trunk_still(cfg): gene_trunk(cfg); gene_still(cfg)
def gene_trunk_fric(cfg): gene_trunk(cfg); gene_fric(cfg)
def gene_trunk_drall(cfg): gene_trunk(cfg); gene_drall(cfg)
def gene_trunk_pc(cfg): gene_trunk(cfg); gene_pc(cfg)
def gene_trunk_fric_pc(cfg): gene_trunk(cfg); gene_fric_pc(cfg)
def gene_trunk_bank(cfg): gene_trunk(cfg); gene_bank(cfg)
def gene_trunk_pivot(cfg): gene_trunk(cfg); gene_pivot(cfg)
def gene_trunk_energy_gated(cfg): gene_trunk(cfg); gene_energy_gated(cfg)
def gene_trunk_energy_cot(cfg): gene_trunk(cfg); gene_energy_cot(cfg)
def gene_trunk_still_alive10(cfg): gene_trunk_still(cfg); gene_alive10(cfg)
def gene_trunk_still_stand15(cfg): gene_trunk_still(cfg); gene_stand15(cfg)
def gene_ctl_clear9(cfg): gene_clear9(cfg)     # on parent_ga: gforce + airtime are already in that parent
