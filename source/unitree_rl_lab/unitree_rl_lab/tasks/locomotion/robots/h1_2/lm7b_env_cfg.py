"""LM7B -- lm7_walls continued, with the torso split baked (2026-10-07).

lm7 sim2sim (2026-10-07, walk rig, fixed reward HUD) named two keepers: the torso SPLIT ("converge to a beautiful 3.00 ... Keeper")
and the balance WALLS ("the gait looks surprisingly better ... This is a keeper"). Every row lost nothing of symonly's walk and none
detonated. What every row still lacks: rotation in place, sideways at vy 0.1, small precise steps. Operator ruling: "rotation and
walking will have to be really precise ... Making tracking even more attractive is the play here."

THE TRUNK = the LM7 stack + gene_walls + gene_split (both from lm7_env_cfg, unchanged), tracking at symonly's 10 / 7 / -1 and the
kernel std at 0.15 so the precision rows isolate their lever. PARENT = milestones/lm7_walls_2026-10-06 (model_70420): the walls block
reshaped the gait, which prices take thousands of iterations to carve; the split is a kernel bonus that climbed from 0 to 22 % inside
one run and the walls policy is already "without much torso shimmying". The upright gene is OUT (the common factor of the two swaying
rows). "yes walls as the parent and then split on top of it. Make sure the split is there."

THE LEVERS (one gene each; rows compose them):
  precision by kernel   gene_std10 / gene_std05: track std 0.15 -> 0.10 / 0.05 on BOTH trackers. Under std 0.15 standing still at a
                        0.10 command keeps exp(-(0.10/0.15)^2) = 64 % of the tracking income (why lm5g's samplers changed nothing);
                        at 0.05 the same error costs 98 %. The far gradient is track_err (L2, already in the ledger).
  precision by price    gene_fw (15/10/-2 = lm7_track's "free win"), gene_track2 (20/14/-2), gene_track4 (40/28/-4).
  rotation in place     gene_spin: spin_prob 0.25 (vx=vy=0, |wz| >= 0.2) + wz box +-0.8 -- ONLY in combination with a precision lever
                        (lm5g_wz, spin alone, REJECTED 09-04).
  low-speed band        gene_slow: slow_prob 0.3 (lm5g_slowdiet, slow alone, REJECTED 09-04) -- crazy row only.
  the A block           gene_lm6gait (gait 3 + feet_slide -2), gene_lm6full (= lm7 gene_lm6: track + turn box + gait3 + slide2).
  prices                gene_energy (-0.01), gene_near22 (feet_too_near 0.18 -> 0.22, crazy row only).
"""
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .lm7_env_cfg import _lm7_stack, gene_walls, gene_split, gene_track, gene_lm6, gene_gait3, gene_slide2, gene_energy  # noqa: F401 (re-exported)

LM7B_PARENT = "lm7_walls_2026-10-06"


def _make_lm7b(cfg):
    gene_walls(cfg)
    gene_split(cfg)
    R = cfg.rewards; C = cfg.commands.base_velocity
    # the split is there
    assert R.torso_stability_bonus is None, "[LM7B] torso product removed"
    assert R.torso_stability_lin.weight == 3.0 and R.torso_stability_lin.params["std_lin"] == 0.15 and R.torso_stability_lin.params["command_name"] == "base_velocity", "[LM7B] torso split lin 3 / .15 cmd-gated"
    assert R.torso_stability_ang.weight == 3.0 and R.torso_stability_ang.params["std_ang"] == 0.30 and R.torso_stability_ang.params["command_name"] == "base_velocity", "[LM7B] torso split ang 3 / .30 cmd-gated"
    # the walls are there
    assert R.action_cap.weight == -0.05 and R.action_cap.params["square"] is True and R.action_cap.params["threshold"] == 1.5, "[LM7B] squared action cap"
    assert R.knee_torque_cap.weight == -0.2 and R.knee_torque_cap.params["limit_nm"] == 100.0, "[LM7B] knee torque cap"
    assert R.feet_too_far.weight == -8.0 and R.feet_too_far.params["threshold"] == 0.60, "[LM7B] feet_too_far -8 @ .60"
    assert R.feet_crossed.weight == -8.0 and R.feet_crossed.func is mdp.feet_crossed_cmd_gated and R.feet_crossed.params["command_name"] == "base_velocity", "[LM7B] feet_crossed cmd-gated"
    assert R.joint_vel_reg.weight == -0.01, "[LM7B] joint_vel_reg"
    # the trunk leaves every lever at its symonly value (rows move them)
    assert R.track_lin_vel_xy.weight == 10.0 and R.track_ang_vel_z.weight == 7.0 and R.track_err.weight == -1.0, "[LM7B] tracking 10/7/-1 in the trunk"
    assert R.track_lin_vel_xy.params["std"] == 0.15 and R.track_ang_vel_z.params["std"] == 0.15, "[LM7B] track std .15 in the trunk"
    assert (C.slow_prob or 0.0) == 0.0 and (C.spin_prob or 0.0) == 0.0 and tuple(C.ranges.ang_vel_z) == (-0.5, 0.5) and C.rel_heading_envs == 0.25, "[LM7B] samplers off, wz box .5 in the trunk"
    assert R.feet_too_near.params["threshold"] == 0.18 and R.gait.weight == 1.5 and R.feet_slide.weight == -0.5 and getattr(R, "energy", None) is None, "[LM7B] near .18, gait 1.5, slide -.5, no energy in the trunk"
    assert getattr(R, "upright_bonus", None) is None or R.upright_bonus.weight in (0.0, 1.0), "[LM7B] no upright gene (the sway suspect)"
    assert tuple(C.ranges.lin_vel_x) == (-0.8, 1.2) and tuple(C.ranges.lin_vel_y) == (-0.3, 0.3), "[LM7B] linear box 1.2"


# ── genes (one lever each; a job composes several via edit_raw lines, applied in order) ──
def gene_std10(cfg):
    cfg.rewards.track_lin_vel_xy.params["std"] = 0.10
    cfg.rewards.track_ang_vel_z.params["std"] = 0.10

def gene_std05(cfg):
    cfg.rewards.track_lin_vel_xy.params["std"] = 0.05
    cfg.rewards.track_ang_vel_z.params["std"] = 0.05

def gene_fw(cfg):          # lm7_track's gene, the "free win" at its tested dose
    gene_track(cfg)

def gene_track2(cfg):
    cfg.rewards.track_lin_vel_xy.weight = 20.0
    cfg.rewards.track_ang_vel_z.weight = 14.0
    cfg.rewards.track_err.weight = -2.0

def gene_track4(cfg):
    cfg.rewards.track_lin_vel_xy.weight = 40.0
    cfg.rewards.track_ang_vel_z.weight = 28.0
    cfg.rewards.track_err.weight = -4.0

def gene_spin(cfg):
    cfg.commands.base_velocity.spin_prob = 0.25
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

def gene_slow(cfg):
    cfg.commands.base_velocity.slow_prob = 0.30

def gene_near22(cfg):
    cfg.rewards.feet_too_near.params["threshold"] = 0.22

def gene_lm6gait(cfg):
    gene_gait3(cfg); gene_slide2(cfg)

def gene_lm6full(cfg):     # the whole lm7_lm6 block on the new trunk
    gene_lm6(cfg)


def _lm7b_stack(cfg):
    _lm7_stack(cfg)
    _make_lm7b(cfg)


@configclass
class RobotEnvCfgLM7B(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _lm7b_stack(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgLM7B(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _lm7b_stack(self)
        _apply_overrides(self, _load_overrides())
