"""LM7U -- lm7_union continued with the lm7b precision setting and the .22 stance (the lm7b_u wave, operator 2026-10-09).

WHY. lm7_union is the line's favourite walk ("The wlaking itself looks really nice and clean i am surprised") with three defects: legs
pitched / rolled inward, no tracking below 0.15 m/s, no rotation in place. The lm7b sim2sim answered the precision question --
"the std + track somehow needs to go hand in hand": std .05 + the 2x price (std05_track2) walks at 0.05 m/s and "is starting to look like
the full package almost" -- and the operator: "Keep precision at std05 ... Stance yes. Union yes. preciison at 2x."

THE TRUNK = the LM7 stack + gene_union (lm6 block: track, turn box +-0.8 / heading .5, gait 3, slide -2; balance block: torso split,
walls, upright_standing) + gene_track2 (20 / 14 / -2) + gene_std05 (both tracking kernels std .05) + gene_near22 (feet_too_near .18 ->
.22, the next notch against the inward legs). No energy in the trunk. Parent milestones/lm7_union_2026-10-07 (model_70420).

GENES (one lever per row):
  energy   gene_e2 / gene_e4 (flat -0.005 / -0.0025), gene_g2 / gene_g4 (energy_cmd_gated: -0.01 at a zero command, x1/2 / x1/4 while
           a velocity is commanded), gene_cot (energy_per_distance: -0.01 at a stand, halved at 0.5 m/s, quartered at 1.0)
  ladder   gene_ladder: tracking_std_levels .15 -> .10 -> .05 at 50 % of the tracking weight (instead of the fixed .05)
  rotation gene_swing (clearance pays only for moving feet), gene_yaw (track_yaw_err_l2 -4), gene_rot (both) -- probe_rotation
           2026-10-09: under a turn command the walkers stand because standing pays 6.2/s more
"""
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .lm7_env_cfg import _lm7_stack, gene_union
from .lm7b_env_cfg import gene_near22, gene_std05, gene_track2

LM7U_PARENT = "lm7_union_2026-10-07"   # model_70420
LM7U_LADDER = (0.15, 0.10, 0.05)
LM7U_LADDER_GATE = 0.5
LM7U_YAW_W = -4.0


def _make_lm7u(cfg):
    gene_union(cfg)
    gene_track2(cfg)
    gene_std05(cfg)
    gene_near22(cfg)
    R = cfg.rewards; C = cfg.commands.base_velocity
    assert R.track_lin_vel_xy.weight == 20.0 and R.track_ang_vel_z.weight == 14.0 and R.track_err.weight == -2.0, "[LM7U] tracking 20 / 14 / -2"
    assert R.track_lin_vel_xy.params["std"] == 0.05 and R.track_ang_vel_z.params["std"] == 0.05, "[LM7U] tracking std .05"
    assert R.feet_too_near.params["threshold"] == 0.22, "[LM7U] feet_too_near .22"
    assert R.torso_stability_bonus is None and R.torso_stability_lin.weight == 3.0 and R.torso_stability_ang.weight == 3.0, "[LM7U] torso split"
    assert R.action_cap.weight == -0.05 and R.feet_too_far.weight == -8.0 and R.feet_crossed.func is mdp.feet_crossed_cmd_gated and R.knee_torque_cap.weight == -0.2, "[LM7U] walls"
    assert R.upright_bonus.func is mdp.upright_bonus_standing and R.upright_bonus.weight == 3.0, "[LM7U] upright_standing (union keeps it)"
    assert R.gait.weight == 3.0 and R.feet_slide.weight == -2.0, "[LM7U] lm6 gait 3 / slide -2"
    assert tuple(C.ranges.ang_vel_z) == (-0.8, 0.8) and C.rel_heading_envs == 0.5, "[LM7U] the turn gene's box and heading share"
    assert R.feet_clearance.func is mdp.foot_clearance_reward_cmd and R.feet_clearance.weight == 12.0, "[LM7U] clearance as the union trained it"
    assert getattr(R, "energy", None) is None and getattr(R, "track_yaw", None) is None, "[LM7U] no energy, no yaw L2 in the trunk"
    assert (C.slow_prob or 0.0) == 0.0 and (C.spin_prob or 0.0) == 0.0, "[LM7U] samplers off"


def _lm7u_stack(cfg):
    _lm7_stack(cfg)
    _make_lm7u(cfg)


# ── genes ──
def _energy(cfg, func, weight, **params):
    cfg.rewards.energy = RewTerm(func=func, weight=weight, params=params)
    assert cfg.rewards.energy.func is func and cfg.rewards.energy.weight == weight


def gene_e2(cfg):
    _energy(cfg, mdp.energy, -0.005)


def gene_e4(cfg):
    _energy(cfg, mdp.energy, -0.0025)


def gene_g2(cfg):
    _energy(cfg, mdp.energy_cmd_gated, -0.01, command_name="base_velocity", moving_scale=0.5, cmd_ref=0.1)


def gene_g4(cfg):
    _energy(cfg, mdp.energy_cmd_gated, -0.01, command_name="base_velocity", moving_scale=0.25, cmd_ref=0.1)


def gene_cot(cfg):
    _energy(cfg, mdp.energy_per_distance, -0.01, command_name="base_velocity", v_floor=0.25, wz_lever=0.5)


def gene_ladder(cfg):
    R = cfg.rewards
    for t in (R.track_lin_vel_xy, R.track_ang_vel_z):
        t.params["std"] = LM7U_LADDER[0]
    cfg.curriculum.track_std = CurrTerm(func=mdp.tracking_std_levels,
                                        params={"reward_terms": ("track_lin_vel_xy", "track_ang_vel_z"), "levels": LM7U_LADDER,
                                                "gate_term": "track_lin_vel_xy", "gate_frac": LM7U_LADDER_GATE})
    assert R.track_lin_vel_xy.params["std"] == 0.15 and cfg.curriculum.track_std.params["levels"] == (0.15, 0.10, 0.05), "[ladder] starts at .15"


def gene_swing(cfg):
    R = cfg.rewards
    R.feet_clearance.func = mdp.foot_clearance_swing_cmd
    assert R.feet_clearance.weight == 12.0 and R.feet_clearance.params["target_height"] == 0.15 and R.feet_clearance.params["command_name"] == "base_velocity", "[swing] same weight / target / gate"


def gene_yaw(cfg):
    cfg.rewards.track_yaw = RewTerm(func=mdp.track_yaw_err_l2, weight=LM7U_YAW_W, params={"command_name": "base_velocity"})
    assert cfg.rewards.track_yaw.weight == -4.0


def gene_rot(cfg):
    gene_swing(cfg)
    gene_yaw(cfg)


@configclass
class RobotEnvCfgLM7U(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _lm7u_stack(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgLM7U(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _lm7u_stack(self)
        _apply_overrides(self, _load_overrides())
