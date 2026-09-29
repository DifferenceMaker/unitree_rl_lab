"""LM6 -- the p14 balance-phase conclusions unioned onto locomotion, trained FROM SCRATCH.

Operator 2026-09-28: "take the latest findings and just union them onto locomotion ... remember
to leave the gating principle on locomotion still ... implement runs and not warmstart them ...
I just want a fresh start." Parent recipe = the LM5-C trunk + the lm5f_symonly union (the last
promoted locomotion parent, re-applied per job until now -> baked here, TRUNK-IS-TASK law).

WHAT CHANGES vs Unitree-H1_2-LM5-C + the symonly union (every item measured on the lm5h ledger,
last 100 it, income as a fraction of its own ceiling, 2026-09-28):
  PLANT   body h1_2_comx06_hand790.urdf (the real 790 g hands, p14b+ trunk body) and the Unitree
          armature table (UNITREE_H1_2_CFG since 2026-09-08: hip pitch/roll + knee 0.1605,
          shoulder 0.0633). Every lm5 run trained at armature 0.01 on the 0.19 kg-hand body;
          p13e/p13f showed the plant alone flips a lean verdict at byte-identical rewards.
  SPLIT   torso_stability_bonus (product, std .07/.15, w6) earned 0.6% = dead -> torso_stability_lin
          (std .15, w3) + torso_stability_ang (std .30, w3), yaw still scored relative to the
          commanded wz (lm5e fix, command_name). Rule-4 widening from 0.6% to the 37% target is a
          x2.3 on both stds, which lands ON the balance trunk's values.
  TILT    upright_bonus std .01 w1 earned 2% = dead cliff -> the p14 LOCKED lever std .05 w3,
          gated on the STANDING command (upright_bonus_standing). flat_orientation_l2 -4.5 stays
          the walking price.
  STANCE  stance_bonus std .15 -> .25 (7% earned = the flat kernel p14c convicted).
          feet_too_near .18 -> .25 (p14b_stance: "Stance solved it"), nominal foot y +-.10 -> +-.13,
          feet_crossed -8 min_gap .05 gated OFF for sideways/turn commands, feet_too_far -8 at 0.60
          (NOT p14's 0.45: the 1.2 m/s stride is ~0.45 m).
  GATE    gait: feet_gait_recovery gated on BASE speed (not in the policy obs) -> feet_gait gated on
          the COMMAND (never gate a reward on unobservable state, p14_recovery_gate 2026-09-15).
  OUT     stride_symmetry (0.0% earned, never differentiated a run) and arm_antiphase (0.0%) are NOT
          carried (operator: "leave it out for now"); stride_track (weight 0 since lm5f) dropped.
          shoulder_roll_swing / the lm5g arms genes are NOT in the trunk: they are the lm6_arms axis.
  KEPT    standing_pose w2 (flat; lm6_nopose is the removal axis), feet_slide -0.5 (symonly's),
          joint_deviation_arms -1.0 incl. shoulder_pitch (lm6_arms frees it), energy ABSENT
          (lm6_energy adds the p14 dose), adaptive lr 1e-3 + mirror 0.1 + GuardedPPO
          (LM4BMirror01PPORunnerCfg; fixed lr is a POLISH tool -- it halves exploration, and
          rotation / low speed are skills still to be learned).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_kitchen import _use_real_hand
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .lm3_env_cfg import _make_lm3
from .lm4_env_cfg import _make_lm4
from .lm4b_env_cfg import _make_lm4b
from .lm5_env_cfg import _make_lm5, _make_lm5d_combo

ARMATURE_HIP_KNEE = 0.160478022   # Unitree table, baked 2026-09-08 -- asserted by lineage_check_lm6.py


def _make_lm5f_symonly(cfg):
    """The lm5f_symonly union, transcribed 1:1 from scripts/queue/lm5f/lm5f_symonly_resume.job
    (= the lm5e_combo union + the 1.2 m/s ceiling), MINUS stride_symmetry (operator 2026-09-28)."""
    cfg.rewards.joint_deviation_hips.func = mdp.joint_deviation_l1_turn_gated
    cfg.rewards.joint_deviation_hips.params["command_name"] = "base_velocity"
    cfg.rewards.joint_deviation_hips.params["wz_ref"] = 0.5
    cfg.rewards.track_lin_vel_xy.params["std"] = 0.15
    cfg.rewards.track_ang_vel_z.params["std"] = 0.15
    cfg.rewards.track_err.weight = -1.0
    cfg.commands.base_velocity.rel_heading_envs = 0.25
    cfg.commands.base_velocity.limit_ranges.ang_vel_z = (-0.8, 0.8)
    cfg.commands.base_velocity.corner_prob = 0.25
    cfg.rewards.track_ang_vel_z.weight = 7.0
    cfg.rewards.flat_orientation_l2.weight = -4.5
    cfg.commands.base_velocity.limit_ranges.lin_vel_x = (-0.8, 1.2)
    cfg.rewards.stride_track = None      # weight 0.0 since lm5f: dead weight, dropped


def _make_lm6(cfg):
    R = cfg.rewards
    # ---- PLANT: real hands (the armature table rides in with UNITREE_H1_2_CFG) ----
    _use_real_hand(cfg)

    # ---- SPLIT torso_stability (p13h_split -> p14 trunk), yaw relative to the wz command ----
    torso = SceneEntityCfg("robot", body_names="torso_link")
    R.torso_stability_bonus = None
    R.torso_stability_lin = RewTerm(
        func=mdp.torso_stability_bonus, weight=3.0,
        params={"std_lin": 0.15, "std_ang": 1000000.0, "command_name": "base_velocity", "asset_cfg": torso},
    )
    R.torso_stability_ang = RewTerm(
        func=mdp.torso_stability_bonus, weight=3.0,
        params={"std_lin": 1000000.0, "std_ang": 0.30, "command_name": "base_velocity", "asset_cfg": torso},
    )

    # ---- TILT: the p14 lever, standing-gated ----
    R.upright_bonus = RewTerm(
        func=mdp.upright_bonus_standing, weight=3.0,
        params={"std": 0.05, "command_name": "base_velocity"},
    )

    # ---- STANCE geometry ----
    R.stance_bonus_legs_torso.params["std"] = 0.25
    R.feet_too_near.params["threshold"] = 0.25
    R.foot_stance_tracking.params["nominal_foot_pos_b"] = [[0.0, 0.13], [0.0, -0.13]]
    feet = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])
    R.feet_too_far = RewTerm(func=mdp.feet_too_far, weight=-8.0, params={"threshold": 0.60, "asset_cfg": feet})
    R.feet_crossed = RewTerm(
        func=mdp.feet_crossed_cmd_gated, weight=-8.0,
        params={"min_gap": 0.05, "command_name": "base_velocity", "vy_max": 0.1, "wz_max": 0.2},
    )

    # ---- GATE on the observable: gait phasing income gated on the command ----
    R.gait = RewTerm(
        func=mdp.feet_gait, weight=R.gait.weight,
        params={"period": 0.75, "offset": [0.0, 0.5], "threshold": 0.55,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )

    # ---- OUT: the never-paying pair is simply not added; nothing else to delete ----
    for dead in ("stride_symmetry", "arm_antiphase", "shoulder_roll_swing"):
        assert getattr(R, dead, None) is None, f"[LM6] {dead} must not be in the trunk"
    assert cfg.scene.robot.spawn.asset_path.endswith("h1_2_comx06_hand790.urdf")


def _lm6_stack(cfg):
    _make_lm3(cfg)
    _make_lm4(cfg)
    _make_lm4b(cfg)
    _make_lm5(cfg)
    _make_lm5d_combo(cfg)
    _make_lm5f_symonly(cfg)
    _make_lm6(cfg)


@configclass
class RobotEnvCfgLM6(RobotEnvCfg):
    """Unitree-H1_2-LM6: scratch trunk, the p14 conclusions on the LM5-C + symonly recipe."""
    def __post_init__(self):
        super().__post_init__()
        _lm6_stack(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgLM6(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _lm6_stack(self)
        self.commands.base_velocity.debug_vis = True
        _apply_overrides(self, _load_overrides())


# ── LM6B: the same trunk with COMPETENCE-gated push curricula (2026-09-29) ──────────────────
# lm6 (clock-stepped pushes): all four rows flat at reward ~200 from iteration 500 to 10000,
# episode length ~530 steps = 10.6 s = the first push, base_height termination 1.00, action std
# frozen at 0.96, lin_vel_levels never left 0.40. push_velocity hit 1.5 m/s by iteration ~1000.
# Operator: "we are administrating the curric level increase too frequently ... sparse out the
# jumps ... implement the ramping the same way unitree does it? They had a gated ramp." /
# "Tracking walking before adding pushes is the right call ... make sure the pushing comes after the
# tracking is fine."
# The gate: pushes advance only when TRACKING is fine (track_lin_vel_xy earns >= 50 % of its weight over
# the recent ~4096 episode ends -- the upstream lin_vel_cmd_levels metric, which couples quality with
# survival) AND >= 50 % of those episodes ran to time-out, >= 1000 it per level, first level held 1000 it.
LM6B_GATE = dict(gate_reward_term="track_lin_vel_xy", gate_frac=0.5, survival_term="time_out", survival_frac=0.5,
                 min_resets=4096, min_hold_steps=24000, warmup_steps=24000)
# The SCRATCH tracking economy (lm4b_mirror01 / lm4c_lcp_scratch, the only scratch walkers that ever escaped the
# first-push plateau): w 3 / 1.5, std 0.5, no track_err. lm6's 10 / 7 at std 0.15 + track_err -1 came from lm5e
# onward, all WARMSTARTED policies that already tracked; at scratch-quality errors (0.3-0.5 m/s) a std .15 kernel
# earns ~nothing and has no gradient (Bible rule 31), and no lm6 row escaped in 10k it. The sharp economy is a
# later polish stage on a warmstart (lm5e "precise"), re-applied by the lm6b_sharp control row.
LM6B_TRACK_W, LM6B_TRACK_ANG_W, LM6B_TRACK_STD = 3.0, 1.5, 0.5


def _make_lm6b(cfg):
    from isaaclab.managers import CurriculumTermCfg as CurrTerm
    R = cfg.rewards
    R.track_lin_vel_xy.weight = LM6B_TRACK_W
    R.track_ang_vel_z.weight = LM6B_TRACK_ANG_W
    R.track_lin_vel_xy.params["std"] = LM6B_TRACK_STD
    R.track_ang_vel_z.params["std"] = LM6B_TRACK_STD
    R.track_err = None
    pv = cfg.curriculum.push_velocity
    cfg.curriculum.push_velocity = CurrTerm(
        func=mdp.push_velocity_curriculum_gated,
        params={"event_term_name": "push_robot", "levels": tuple(pv.params["levels"]), **LM6B_GATE},
    )
    sp = cfg.curriculum.sustained_push
    cfg.curriculum.sustained_push = CurrTerm(
        func=mdp.sustained_push_curriculum_gated,
        params={"event_term_name": "sustained_push_apply", "levels": tuple(sp.params["levels"]), **LM6B_GATE},
    )
    assert cfg.curriculum.push_velocity.params["levels"][-1] == 1.5, "[LM6B] push ceiling must stay 1.5 m/s"
    assert cfg.curriculum.push_velocity.params["gate_reward_term"] == "track_lin_vel_xy", "[LM6B] pushes gate on tracking"
    assert R.track_lin_vel_xy.params["std"] == 0.5 and R.track_lin_vel_xy.weight == 3.0 and R.track_err is None, "[LM6B] scratch tracking economy"


@configclass
class RobotEnvCfgLM6B(RobotEnvCfg):
    """Unitree-H1_2-LM6B: LM6 + survival-gated push / sustained-push curricula."""
    def __post_init__(self):
        super().__post_init__()
        _lm6_stack(self)
        _make_lm6b(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgLM6B(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _lm6_stack(self)
        _make_lm6b(self)
        self.commands.base_velocity.debug_vis = True
        _apply_overrides(self, _load_overrides())
