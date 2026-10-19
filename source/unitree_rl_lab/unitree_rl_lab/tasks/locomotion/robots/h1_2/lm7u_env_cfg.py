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
from isaaclab.managers import SceneEntityCfg
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


# ── lm7c (2026-10-12, parent lm7b_u_ladder @77419; the ladder gene is re-applied by every row, the level re-climbs on resume) ──
def gene_g4_yaw(cfg):
    gene_g4(cfg)
    gene_yaw(cfg)


def gene_clear6(cfg):
    """feet_clearance 12 -> 6, target / std / gate kept: a walk that follows the clearance less (operator 2026-10-12 on the swing
    rows' lift: "makslig" -- artificial). Works on the plain term (lm7c_clear6) and on the swing form (lm7c_rot_clear6)."""
    R = cfg.rewards
    assert R.feet_clearance.weight == 12.0 and R.feet_clearance.params["target_height"] == 0.15, "[clear6] from the trunk's 12 @ .15"
    R.feet_clearance.weight = 6.0


def gene_rot_clear6(cfg):
    gene_rot(cfg)
    gene_clear6(cfg)


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


# ── lm7d (2026-10-19): base = the ladder + the PLAIN clearance at 6 (lm7c_clear6, the operator's gait winner), parent
# lm7c_clear6 @84400; one lever per row, the limit / inward-leg axes also on top of dead03 and torsowalk, two turn combos.
_FEET = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])
_GATED_CMD_THR = ("torso_lin_vel_xy", "torso_ang_vel", "base_pos_xy_l2_from_spawn", "standing_pose", "feet_clearance", "gait")


def gene_lm7d_base(cfg):
    gene_ladder(cfg)
    gene_clear6(cfg)


def gene_yaw12(cfg):
    """track_yaw_err_l2 at -12: the small-turn slope (rot at -4 only turned above ~0.35 rad/s commanded -- a 0.3 error
    priced 0.36/s loses to the standing incomes; at -12 it costs 1.08/s)."""
    cfg.rewards.track_yaw = RewTerm(func=mdp.track_yaw_err_l2, weight=-12.0, params={"command_name": "base_velocity"})


def _swing_bonus(cfg, target):
    """A SMALL moving-foot clearance bonus next to the plain clear6 (not instead of it): the probe's turn mechanism (a
    lifted foot earns, so stepping beats twisting the hip) at a LOW target -- the 'artificial' lift of the swing rows was the
    0.15 m target reached by a fast short step (G1 upstream: 0.10)."""
    assert cfg.rewards.feet_clearance.weight == 6.0 and getattr(cfg.rewards, "feet_swing", None) is None, "[turn] on clear6, once"
    cfg.rewards.feet_swing = RewTerm(func=mdp.foot_clearance_swing_cmd, weight=3.0,
                                     params={"asset_cfg": _FEET, "target_height": target, "std": 0.05, "tanh_mult": 2.0, "command_name": "base_velocity"})


def gene_turn(cfg):
    _swing_bonus(cfg, 0.10)
    gene_yaw(cfg)


def gene_turn05(cfg):
    _swing_bonus(cfg, 0.05)
    gene_yaw(cfg)


def gene_dead03(cfg, thr=0.03):
    """Every command gate 0.1 -> 0.03, standing side and walking side (the sampler is untouched: its standing envs are
    exact zeros). A 0.05 command is 'stand' to half the ledger today: stillness prices + standing bonuses ON, clearance +
    gait OFF. The keys window steps in 0.05, so the threshold must sit BELOW 0.05 (a strict <, > pair)."""
    R = cfg.rewards
    for n in _GATED_CMD_THR:
        getattr(R, n).params["cmd_thr"] = thr
    assert R.upright_bonus.func is mdp.upright_bonus_standing
    R.upright_bonus.params["thr"] = thr
    for opt in ("feet_swing", "torso_ang_vel_walk"):          # the turn / torsowalk rows, when composed before this gene
        t = getattr(R, opt, None)
        if t is not None:
            t.params["cmd_thr"] = thr


def gene_torsowalk(cfg):
    """Torso angular-velocity L2 WHILE WALKING at -0.5 (the standing term stays at -2): the operator's low-speed torso axis."""
    cfg.rewards.torso_ang_vel_walk = RewTerm(func=mdp.body_ang_vel_l2_walking, weight=-0.5,
                                             params={"command_name": "base_velocity", "cmd_thr": 0.1, "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])})


def gene_dofcap(cfg):
    """dof_pos_limits -100 -> -400: lm7c_rot_clear6 rode the hip-yaw stop for -4.3/s and still won (Bible rule 33: a price
    that is PAID is too cheap to be a wall)."""
    assert cfg.rewards.dof_pos_limits.weight == -100.0, "[dofcap] from the trunk's -100"
    cfg.rewards.dof_pos_limits.weight = -400.0


def gene_near26(cfg):
    """feet_too_near .22 -> .26: the inward legs (cont / seed / yaw / rot all 'legs move inward')."""
    assert cfg.rewards.feet_too_near.params["threshold"] == 0.22, "[near26] from the trunk's .22"
    cfg.rewards.feet_too_near.params["threshold"] = 0.26


def gene_footyaw(cfg):
    """feet_yaw_l2_turn_gated -10: foot orientation priced for the first time (the outward-rotated leg / default stance).
    -1 was INERT in the 90-it liveness (-0.005/s; a 20-degree outward foot on both sides would cost 0.24/s against a 66/s
    ledger); at -10 it is 2.4/s at 20 deg both / 0.6/s at 10 deg -- a price, not a wall."""
    cfg.rewards.feet_yaw = RewTerm(func=mdp.feet_yaw_l2_turn_gated, weight=-10.0,
                                   params={"asset_cfg": _FEET, "command_name": "base_velocity", "wz_ref": 0.5})


def gene_dead03_dofcap(cfg): gene_dead03(cfg); gene_dofcap(cfg)
def gene_dead03_near26(cfg): gene_dead03(cfg); gene_near26(cfg)
def gene_dead03_footyaw(cfg): gene_dead03(cfg); gene_footyaw(cfg)
def gene_torsowalk_dofcap(cfg): gene_torsowalk(cfg); gene_dofcap(cfg)
def gene_torsowalk_near26(cfg): gene_torsowalk(cfg); gene_near26(cfg)
def gene_torsowalk_footyaw(cfg): gene_torsowalk(cfg); gene_footyaw(cfg)
def gene_turn_dead03(cfg): gene_turn(cfg); gene_dead03(cfg)        # dead03 LAST so feet_swing's gate follows
def gene_turn_torsowalk(cfg): gene_turn(cfg); gene_torsowalk(cfg)
