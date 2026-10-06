"""LM7 -- the lm5f_symonly walker, continued (2026-10-06).

WHY (operator, 2026-10-06, after the lm6c previews/sim2sim): the lm6 line was SCRATCH on a stander's ledger
and marched in place; lm5f_symonly (model_63421, 63k it of warmstarted walking) "walks forward, backward
and sideways", tracks 57 % of its weight (lm6c_track 40 % of 15), slides less (raw 0.16 vs 0.22) and falls
less (99.1 % time-outs). What it lacks -- rotation in place, gait, tracking -- is exactly what lm6 worked on.
So the lm6 conclusions and the general-balance conclusions are applied TO THE WALKER, step by step, in
three categories (A: symonly + lm6, B: symonly + balance, C: the union), every row a warmstart from
`milestones/lm5f_symonly_resume_2026-08-31`.

DETONATION: the walk chain (adaptive lr 1e-3 + mirror 0.1, GuardedPPO since 09-05) never detonated in
five warmstarts from 30k to 68k (value loss max 3.6-6.6 after +300, std flat 0.61-0.63); the balance
line's explosions were fixed-lr income deltas. Procedure anyway = the p14e stack: --resume_curriculum
--critic_warmup_iters 300 use_clipped_value_loss=false; fixed lr NOT ported; no std reset (0.61 is healthy).

THE TRUNK (every row): symonly's ledger = LM5-C + `_make_lm5f_symonly` (its job union, which never had
stride_symmetry in the cfg -- the job added it; it earned 0.003/step and carries the 09-04 gate bug, so it
stays out per the 09-28 ruling), plus
  * body h1_2_comx06_hand790 (real 790 g hands; the Unitree motor table rides in with UNITREE_H1_2_CFG),
  * gait gated on the COMMAND (`feet_gait`) instead of on base speed (`feet_gait_recovery` gate_speed .15 =
    a gate on state the policy cannot observe),
  * the linear command box resumed at the parent's FINAL level (lin_vel_x (-0.8, 1.2), lin_vel_y (-0.3,
    0.3) = lin_vel_levels 1.2 at the pick); the yaw box stays symonly's (-0.5, 0.5) except in the turn rows,
  * torso_stability_bonus.command_name = base_velocity restored (symonly's job had it; the lm6 transcription
    dropped it -- the lineage check caught the miss on the first smoke).
Stance geometry stays symonly's (nominal +-0.10, feet_too_near .18, stance std .15): the LM6 widening
(+-.13 / .25 / .25 against 17.5 cm hip spacing) produced the split stance.

GENES (jobs call them through edit_raw so each row is one line and the smoke env.yaml is the proof):
  gene_track   track_lin 10 -> 15, track_ang 7 -> 10, track_err -1 -> -2        (lm6c_track)
  gene_turn    track_ang 7 -> 10, rel_heading_envs .25 -> .5, yaw box (-0.8, 0.8) from the start
               (the 0.5 gate needs 50 % of a kernel that earns 10 %: symonly trained its whole life at +-0.5)
  gene_gait3   gait 1.5 -> 3.0                                                   (lm6c_gait)
  gene_slide2  feet_slide -0.5 -> -2.0                                           (lm6c)
  gene_split   torso_stability product 6.0 (.07/.15, 0.017/step = dead) -> lin 3 (std .15) + ang 3 (std .30,
               yaw vs the commanded wz)                                          (p13h -> p14 trunk)
  gene_walls   action_cap -0.05 squared thr 1.5, knee_torque_cap -0.2 @100 Nm, feet_too_far -8 @0.60,
               feet_crossed -8 command-gated, joint_vel_reg -0.01                (p14 carry list, gait-neutral)
  gene_upright upright 1.0/.01 -> 3.0/.05 STANDING-gated                         (p14f/g lean lever)
  gene_energy  energy -0.01                                                      (p14 dose; both lm6c energy rows were the worst -- union_energy only)
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_kitchen import _use_real_hand
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .lm5_env_cfg import _make_lm3, _make_lm4, _make_lm4b, _make_lm5, _make_lm5d_combo
from .lm6_env_cfg import _make_lm5f_symonly

LM7_PARENT = "lm5f_symonly_resume_2026-08-31"   # model_63421
LM7_LIN_BOX_X = (-0.8, 1.2)
LM7_LIN_BOX_Y = (-0.3, 0.3)
_FEET = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])
_TORSO = SceneEntityCfg("robot", body_names="torso_link")


def _make_lm7(cfg):
    R = cfg.rewards
    _use_real_hand(cfg)
    R.gait = RewTerm(
        func=mdp.feet_gait, weight=R.gait.weight,
        params={"period": 0.75, "offset": [0.0, 0.5], "threshold": 0.55, "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )
    cfg.commands.base_velocity.ranges.lin_vel_x = LM7_LIN_BOX_X
    cfg.commands.base_velocity.ranges.lin_vel_y = LM7_LIN_BOX_Y
    # symonly's product torso term scores YAW RELATIVE TO THE COMMANDED wz (its job set command_name; the lm6
    # transcription dropped it because LM6 replaced the product) -- restored, else every turn is taxed by w 6
    R.torso_stability_bonus.params["command_name"] = "base_velocity"
    assert getattr(R, "stride_symmetry", None) is None and getattr(R, "arm_antiphase", None) is None, "[LM7] dead terms out"
    assert cfg.scene.robot.spawn.asset_path.endswith("h1_2_comx06_hand790.urdf"), "[LM7] hand790 body"
    assert R.gait.func is mdp.feet_gait and R.gait.params["command_name"] == "base_velocity", "[LM7] gait on the command"
    assert R.track_lin_vel_xy.weight == 10.0 and R.track_lin_vel_xy.params["std"] == 0.15 and R.track_err.weight == -1.0, "[LM7] symonly economy"
    assert R.feet_too_near.params["threshold"] == 0.18 and R.stance_bonus_legs_torso.params["std"] == 0.15, "[LM7] symonly stance geometry"
    assert tuple(cfg.commands.base_velocity.ranges.ang_vel_z) == (-0.5, 0.5) and cfg.commands.base_velocity.rel_heading_envs == 0.25, "[LM7] symonly yaw box"
    assert R.torso_stability_bonus.weight == 6.0 and R.feet_slide.weight == -0.5 and R.gait.weight == 1.5, "[LM7] symonly weights"
    assert R.torso_stability_bonus.params.get("command_name") == "base_velocity", "[LM7] torso product scores yaw vs the wz command (symonly)"


# ---------------- genes (edit_raw one-liners) ----------------
def gene_track(cfg):
    cfg.rewards.track_lin_vel_xy.weight = 15.0
    cfg.rewards.track_ang_vel_z.weight = 10.0
    cfg.rewards.track_err.weight = -2.0


def gene_turn(cfg):
    cfg.rewards.track_ang_vel_z.weight = 10.0
    cfg.commands.base_velocity.rel_heading_envs = 0.5
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)


def gene_gait3(cfg):
    cfg.rewards.gait.weight = 3.0


def gene_slide2(cfg):
    cfg.rewards.feet_slide.weight = -2.0


def gene_lm6(cfg):
    gene_track(cfg); gene_turn(cfg); gene_gait3(cfg); gene_slide2(cfg)


def gene_split(cfg):
    R = cfg.rewards
    R.torso_stability_bonus = None
    R.torso_stability_lin = RewTerm(func=mdp.torso_stability_bonus, weight=3.0,
                                    params={"std_lin": 0.15, "std_ang": 1000000.0, "command_name": "base_velocity", "asset_cfg": _TORSO})
    R.torso_stability_ang = RewTerm(func=mdp.torso_stability_bonus, weight=3.0,
                                    params={"std_lin": 1000000.0, "std_ang": 0.30, "command_name": "base_velocity", "asset_cfg": _TORSO})


def gene_walls(cfg):
    R = cfg.rewards
    R.action_cap = RewTerm(func=mdp.action_magnitude_over, weight=-0.05, params={"threshold": 1.5, "square": True, "max_excess": 5.0})
    R.knee_torque_cap = RewTerm(func=mdp.joint_torque_over_limit, weight=-0.2,
                                params={"limit_nm": 100.0, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])})
    R.feet_too_far = RewTerm(func=mdp.feet_too_far, weight=-8.0, params={"threshold": 0.60, "asset_cfg": _FEET})
    R.feet_crossed = RewTerm(func=mdp.feet_crossed_cmd_gated, weight=-8.0,
                             params={"min_gap": 0.05, "command_name": "base_velocity", "vy_max": 0.1, "wz_max": 0.2})
    R.joint_vel_reg = RewTerm(func=mdp.joint_vel_l2, weight=-0.01, params={})


def gene_upright(cfg):
    cfg.rewards.upright_bonus = RewTerm(func=mdp.upright_bonus_standing, weight=3.0,
                                        params={"std": 0.05, "command_name": "base_velocity"})


def gene_balance(cfg):
    gene_split(cfg); gene_walls(cfg); gene_upright(cfg)


def gene_union(cfg):
    gene_lm6(cfg); gene_balance(cfg)


def gene_energy(cfg):
    cfg.rewards.energy = RewTerm(func=mdp.energy, weight=-0.01, params={})


def _lm7_stack(cfg):
    _make_lm3(cfg); _make_lm4(cfg); _make_lm4b(cfg); _make_lm5(cfg); _make_lm5d_combo(cfg)
    _make_lm5f_symonly(cfg)
    _make_lm7(cfg)


@configclass
class RobotEnvCfgLM7(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _lm7_stack(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgLM7(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _lm7_stack(self)
        _apply_overrides(self, _load_overrides())
