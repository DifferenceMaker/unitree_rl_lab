"""lm7l (2026-11-05): the swing ARCH + the arm sway, on lm7k_micro_legs_squeeze @146200 (authored; NOT enqueued -- operator: "don't enqueue anything").

Operator 2026-11-05 sim2sim of lm7k_micro_legs_squeeze ("Looks really good. Looks stable too with almost no swaying. Tracks fine. No immediate
defects besides the arms." GREENLIT): "the legs get raised upwards, instead of upwards and forward if walking forward or upwards and backwards
when walking backwards ... Reward it for following the vector of velocity with feet? So that there is a nice, natural arch to the next step
... we don't want the arch to be too excessive and lunge forward." and "We will definitely want the arms swaying and being like pendulums gone."

Genes (scripts/queue/lm7l/HYPOTHESIS.md); every row = lm7k_env_cfg.lm7k_micro_legs_squeeze (the parent's chain: combo + ffarms + roll +
stopramp2 + clear10 + micro_push + legs + squeeze) + its lever:
  arch      feet_swing.func foot_clearance_swing_cmd -> mdp.foot_clearance_swing_along_cmd (same target .10, std .01, tanh_mult 2, w 12):
            the speed factor tanh(2 |v_xy|) -> tanh(2 max(0, v_foot . u_cmd)), u = the commanded planar direction in the base frame. ONE term,
            not two: the swing reward itself pays only the along-command part of the swing, so an outward arc or a backward kick earns no
            lift income; tanh saturates near 1 m/s so a lunge pays no more than a stride; a pure turn (|v_cmd| < 0.05) keeps the old factor.
  armsway   lm7j gene 5: arm_motion_walking (8 upper-arm DOFs' joint_vel^2 while walking) x -0.25 -- authored here on the new parent so the
            arm axis rides with the arch; the lm7k_armsway row on clear10 (H200) reads the same lever on the old parent.
  arch_add  the same along-command factor as a SECOND term beside the unchanged feet_swing: + swing_along = foot_clearance_swing_along_cmd
            (target .10, std .01, tanh 2, lin_thr .05) x 6 (operator 2026-11-05: "a 2 axis run ... 2 separate runs to test this theory" --
            replace vs add: the added term leaves every swing its 12 and pays up to 6 more for the along-command part)
  arch_armsway  both (arch + armsway).
  slide     lm7k keeper (operator 2026-11-05: "It slides a lot less ... This is a keeper. This goes into the next batch."): feet_slide -2 -> -8
  ramp2     lm7k ramp2 ("controlled and easy ... worth carrying over as a separate axis run on top of the next one"): slew_acc (0.5, 0.5, 1.0)
  upright6  lm7k upright6 ("I like it ... worth as an axis on the next parent of the greenlit policies"): upright_bonus 3 -> 6
  kit       armsway + slide + ramp2 (the three the operator wants carried; rule 36: read against the singles)
  full      arch + armsway + slide + ramp2 + upright6 (operator 2026-11-05: "I propose arch + armsway + slide + ramp2 + upright. Basically the
            full combo. One extra run."; rule 36: the largest stack of the wave = the likeliest detonation, read against kit and the singles)
"""
from isaaclab.managers import RewardTermCfg as RewTerm

from unitree_rl_lab.tasks.locomotion import mdp

from .lm7j_env_cfg import gene_armsway
from .lm7k_env_cfg import lm7k_micro_legs_squeeze, gene_slide, gene_acc, gene_upright6

PARENT, IT = "lm7k_micro_legs_squeeze_2026-11-05", 146200   # stopped at 146216 for the colleague's server; the latest clean window
ARCH_LIN_THR = 0.05   # |v_cmd_xy| below this = a pure turn: the old any-direction speed factor applies (the turn's clearance unchanged)
ARCH_ADD_W = 6.0      # the added along-command bonus: half the swing term's 12 (a perfect along-command swing earns 18, an arc 12)


def parent(cfg):
    """The parent's gene chain, bit-identical (JOB_VALIDITY 3)."""
    lm7k_micro_legs_squeeze(cfg)


def gene_arch(cfg):
    s = cfg.rewards.feet_swing
    assert s.func is mdp.foot_clearance_swing_cmd and s.weight == 12.0 and s.params["target_height"] == 0.10 and s.params["std"] == 0.01 \
        and s.params["tanh_mult"] == 2.0 and s.params["cmd_thr"] == 0.03, "[arch] the parent's feet_swing = foot_clearance_swing_cmd target .10 std .01 tanh 2 w 12"
    s.func = mdp.foot_clearance_swing_along_cmd
    s.params["lin_thr"] = ARCH_LIN_THR


def gene_arch_add(cfg):
    R = cfg.rewards
    assert getattr(R, "swing_along", None) is None and R.feet_swing.func is mdp.foot_clearance_swing_cmd, "[arch_add] the parent's feet_swing untouched, no swing_along underneath"
    R.swing_along = RewTerm(func=mdp.foot_clearance_swing_along_cmd, weight=ARCH_ADD_W,
                            params={**{k: v for k, v in R.feet_swing.params.items()}, "lin_thr": ARCH_LIN_THR})


def lm7l_arch(cfg): parent(cfg); gene_arch(cfg)
def lm7l_arch_add(cfg): parent(cfg); gene_arch_add(cfg)
def lm7l_armsway(cfg): parent(cfg); gene_armsway(cfg)
def lm7l_arch_armsway(cfg): parent(cfg); gene_arch(cfg); gene_armsway(cfg)
def lm7l_slide(cfg): parent(cfg); gene_slide(cfg)
def lm7l_ramp2(cfg): parent(cfg); gene_acc(cfg)
def lm7l_upright6(cfg): parent(cfg); gene_upright6(cfg)
def lm7l_kit(cfg): parent(cfg); gene_armsway(cfg); gene_slide(cfg); gene_acc(cfg)
def lm7l_full(cfg): parent(cfg); gene_arch(cfg); gene_armsway(cfg); gene_slide(cfg); gene_acc(cfg); gene_upright6(cfg)
