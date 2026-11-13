"""study_s1 (2026-11-09): the Optuna study's parametrized gene on the lm7l_arch_armsway chain (operator: "Start implementing Optuna and the study").

Center = milestone lm7l_arch_armsway_2026-11-05 (model_153199; Exam #2 v4 anchor 57.0; operator sim2sim 11-09: "Its gait when walking is
precisely what i want"). Its tracking ledger: track_lin_vel_xy 20 (lm7i track_lin_vel_xy_split, std .15 -> the ladder -> .05), track_ang_vel_z 14,
track_err -2 (track_vel_err_l2), track_yaw -4 (lm7u track_yaw_err_l2). The lm7l land finding (optuna-study-design): the exp kernel is FLAT
beyond ~2 sigma once the ladder reaches .05 -- at e = 0.2 m/s, sigma = .05 the gradient is d/de 20 exp(-e^2/sigma^2) = -2 x 20 x 0.2/0.0025 x
exp(-16) ~ -1.8e-5 per m/s -- so 11/17 lm7l rows stopped walking. The s1 space is the "kernel-near income + far-field gradient" family:

  track_w      track_lin_vel_xy.weight                      center 20    range 10..40   (log)
  track_err_w  track_err.weight (L2 far-field gradient)     center -2    range -8..-0.5 (log on |w|)
  std_floor    the ladder's FINAL std: levels (.15, max(.10, floor), floor), de-duplicated   center .05  range .05..0.15
  guard_w      + track_lin_guard = the SAME kernel function at a FIXED std .15 (never on the ladder)   center 0 (absent)  range 0..10
  yaw_err_w    track_yaw.weight (the yaw L2)                center -4    range -12..-1  (log on |w|)
  enc_scale    measured-residual encoder DR: per-joint band = enc_scale x RESID[j]   center 0 (not applied)  range 0..1.5

Every lever at its center leaves the cfg BIT-IDENTICAL to lm7l_arch_armsway (lineage_check_study.py proves the env.yaml diff is empty).

Bible pre-flight (track_lin_guard): income-near + gradient-far pair -- the ladder's .05 kernel pays the near field, the fixed .15 kernel keeps a
gradient alive out to ~0.3 m/s (d/de guard_w exp(-e^2/.0225) at e = 0.2 is -guard_w x 17.8 x 0.169 = -3.0 guard_w per m/s, i.e. 5 orders above
the .05 kernel's). Gated only on the COMMAND (observed; the split function's std_stand branch rides along as in the main term). On a warmstart
guard_w > 0 adds standing + near-field income at once = a detonation watch (the critic sees a level shift); 0..10 keeps it under the main 20.
enc_scale: the p16c_sw_enc01 mechanism (ArmGravityFFEncJointPositionAction + joint_pos_rel_encoder + the reset event); the band shape is the
ROBOT's measured residual per joint instead of one +-0.01 band (enc01 covered 1/3 of the 3.5 deg ankle-pitch gap; enc_scale 1.0 = the measured
residuals, 1.5 = a 50 % margin). Critic unchanged (true angles) -> warmstart-compatible, no width change.
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.locomotion import mdp

from .lm7l_env_cfg import lm7l_arch_armsway

CENTER_MILESTONE, CENTER_IT = "lm7l_arch_armsway_2026-11-05", 153199
LADDER_TOP, LADDER_MID = 0.15, 0.10          # the parent's ladder (0.15, 0.10, 0.05): the top two rungs stay
GUARD_STD = 0.15                             # the guard kernel's fixed std = the ladder's starting rung
# The robot's measured encoder residuals (rad) per joint group (stop tests 10-07 / 11-06: ankle pitch 1.75 deg, hip roll 1.15, ankle roll 0.75,
# the rest below 0.3 deg; the torso had none). Band for joint j at enc_scale s = +-s x RESID[j].
RESID = {
    ".*_ankle_pitch_joint": 0.0305,
    ".*_hip_roll_joint": 0.020,
    ".*_ankle_roll_joint": 0.013,
    ".*_hip_yaw_joint": 0.005,
    ".*_hip_pitch_joint": 0.004,
    ".*_knee_joint": 0.004,
    "torso_joint": 0.0,
}
RESID_JOINTS = list(RESID.keys())            # the 13 leg + torso joints (== lm7l SW_ENC01_JOINTS == DR_LEG_JOINTS), as 7 group patterns
CENTER = dict(track_w=20.0, track_err_w=-2.0, std_floor=0.05, guard_w=0.0, yaw_err_w=-4.0, enc_scale=0.0)


def ladder_levels(std_floor: float) -> tuple:
    """(0.15, max(0.10, floor), floor) with duplicates removed, as a tuple (the parent stores a tuple): .05 -> (0.15, 0.1, 0.05) == the
    parent's; .10 -> (0.15, 0.10); .15 -> (0.15,) = no narrowing at all."""
    lv = [LADDER_TOP, max(LADDER_MID, float(std_floor)), float(std_floor)]
    out = []
    for x in lv:
        if not out or abs(out[-1] - x) > 1e-9:
            out.append(x)
    return tuple(out)


def gene_track_w(cfg, track_w: float):
    t = cfg.rewards.track_lin_vel_xy
    assert t.func is mdp.track_lin_vel_xy_split and t.weight == 20.0, "[study] the parent's track_lin_vel_xy = split kernel at 20"
    t.weight = float(track_w)


def gene_track_err_w(cfg, track_err_w: float):
    t = cfg.rewards.track_err
    assert t.func is mdp.track_vel_err_l2 and t.weight == -2.0, "[study] the parent's track_err = track_vel_err_l2 at -2"
    assert track_err_w < 0, "[study] track_err_w is a price (negative)"
    t.weight = float(track_err_w)


def gene_std_floor(cfg, std_floor: float):
    c = cfg.curriculum.track_std
    assert c.func is mdp.tracking_std_levels and tuple(c.params["levels"]) == (0.15, 0.10, 0.05) and c.params["gate_term"] == "track_lin_vel_xy" \
        and c.params["gate_frac"] == 0.5 and tuple(c.params["reward_terms"]) == ("track_lin_vel_xy", "track_ang_vel_z"), "[study] the parent's ladder (0.15, 0.10, 0.05) gated at 50 %"
    assert 0.05 - 1e-9 <= std_floor <= 0.15 + 1e-9, "[study] std_floor in [0.05, 0.15]"
    c.params["levels"] = ladder_levels(std_floor)
    # the terms' std stays the parent's starting value (.15 = levels[0]); the curriculum re-sets it from levels on its first call anyway


def gene_guard(cfg, guard_w: float):
    """+ track_lin_guard = track_lin_vel_xy's function and params with std fixed at GUARD_STD, NOT in the ladder's reward_terms."""
    R = cfg.rewards
    assert getattr(R, "track_lin_guard", None) is None, "[study] no track_lin_guard underneath"
    assert "track_lin_guard" not in cfg.curriculum.track_std.params["reward_terms"], "[study] the guard is never on the ladder"
    assert guard_w > 0, "[study] gene_guard only at guard_w > 0 (0 = absent, bit-identical)"
    src = R.track_lin_vel_xy
    R.track_lin_guard = RewTerm(func=src.func, weight=float(guard_w), params={**dict(src.params), "std": GUARD_STD})


def gene_yaw_err_w(cfg, yaw_err_w: float):
    t = cfg.rewards.track_yaw
    assert t.func is mdp.track_yaw_err_l2 and t.weight == -4.0, "[study] the parent's track_yaw = track_yaw_err_l2 at -4"
    assert yaw_err_w < 0, "[study] yaw_err_w is a price (negative)"
    t.weight = float(yaw_err_w)


def gene_enc_resid(cfg, enc_scale: float):
    """gene_enc's mechanism with the per-joint measured-residual band: joint j draws uniform +-enc_scale x RESID[j] per reset."""
    assert enc_scale > 0, "[study] gene_enc_resid only at enc_scale > 0 (0 = not applied, bit-identical)"
    A = cfg.actions.JointPositionAction
    assert type(A).__name__ == "ArmGravityFFJointPositionActionCfg" and A.scale == 0.25 and A.arm_gravity_ff is True, "[study] the parent's arm-ff action term underneath"
    assert cfg.observations.policy.joint_pos_rel.func is mdp.joint_pos_rel and cfg.observations.critic.joint_pos_rel.func is mdp.joint_pos_rel, "[study] clean joint_pos_rel underneath"
    assert getattr(cfg.events, "encoder_offset", None) is None, "[study] no encoder event underneath"
    cfg.actions.JointPositionAction = mdp.arm_gravity_ff_enc_action_cfg(A)
    cfg.observations.policy.joint_pos_rel.func = mdp.joint_pos_rel_encoder      # the policy sees the encoder frame; the critic the true one
    band = {k: round(float(enc_scale) * v, 6) for k, v in RESID.items()}
    cfg.events.encoder_offset = EventTerm(func=mdp.randomize_encoder_offset, mode="reset",
                                          params={"offset_range_per_joint": band,
                                                  "asset_cfg": SceneEntityCfg("robot", joint_names=list(RESID_JOINTS))})


def study_s1(cfg, track_w=20.0, track_err_w=-2.0, std_floor=0.05, guard_w=0.0, yaw_err_w=-4.0, enc_scale=0.0):
    """lm7l_arch_armsway(cfg) + the six s1 levers. At CENTER every lever is a no-op (the gene functions with a 'not applied' center are
    skipped; the weight genes assign the parent's own value) -> env.yaml identical to the center milestone's."""
    lm7l_arch_armsway(cfg)
    gene_track_w(cfg, track_w)
    gene_track_err_w(cfg, track_err_w)
    gene_std_floor(cfg, std_floor)
    if float(guard_w) > 0.0:
        gene_guard(cfg, guard_w)
    gene_yaw_err_w(cfg, yaw_err_w)
    if float(enc_scale) > 0.0:
        gene_enc_resid(cfg, enc_scale)


# ══ study s2 (2026-11-13) ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
# Operator 2026-11-13: "First train the winner of the study with the encoders AND then put it as the next study's parent and make a
# study to find what we did with s1 but now with vy and wz too" / "this is purely hyperparameter hunt for a profitable reward arithmetic
# for walking at low speeds vs standing" / "We need to be surgical" / "we shouldn't be too cautious about the hyperparameter limits".
# The s1 top trials track above a threshold and STAND below it (vx, vy +-0.05 / +-0.10; |wz| <= 0.25-0.35 rad/s). s2 = the PARENT's
# cfg (an lm7p row: study_s1 levers + the encoder band) with
#   * the band HELD at +-0.010 rad in every trial (no ladder; not a lever),
#   * std_floor and guard_w FIXED at the parent's values,
#   * EIGHT levers:
#       track_w     track_lin_vel_xy weight (s1 ended on its lower bound)
#       vx_err_w    } track_err (the squared price on the SUM of the vx, vy, wz errors) SPLIT into one price per axis:
#       vy_err_w    } track_vx / track_vy = mdp.track_vx_err_l2 / track_vy_err_l2, and the yaw part goes into track_yaw
#       yaw_err_w   track_yaw weight (the existing squared yaw price, now carrying track_err's yaw share too)
#       lin_l1_w    + track_lin_l1 = mdp.track_lin_err_l1, the absolute-error price on the planar velocity (0 = absent)
#       yaw_l1_w    + track_yaw_l1 = mdp.track_yaw_err_l1, the absolute-error price on the turn rate (0 = absent)
#       feet_yaw_w  feet_yaw_bonus weight: it pays feet parallel to the body and fades out only at a 0.5 rad/s turn command, so at
#                   a 0.2 rad/s command it is 60 % on and pays AGAINST stepping round (the turn deadband ends where it fades out)
#       torso_rel   torso_stability_lin scored relative to the command: 0 = the parent's absolute torso speed (a perfect 0.10 m/s
#                   walk forfeits 36 % of the bonus), 1 = only the speed that differs from the command (mdp.torso_stability_lin_rel)
# At the center (vx_err_w = vy_err_w = the parent's track_err_w, yaw_err_w = the parent's yaw_err_w + its track_err_w, the L1 prices
# 0, feet_yaw_w 3, torso_rel 0) every step pays exactly what the parent's cfg pays; the terms are split, the totals are not changed.
# The parent is plugged in by S2_PARENT (its fixed values) + the space file's center (its milestone and lever values).
from .lm7l_env_cfg import SW_ENC01_JOINTS, gene_enc   # noqa: E402

S2_BAND = 0.010          # rad: p16c_sw_enc01's band = the ladder's top rung, held
S2_PARENTS = {           # the lm7p rows (aspired-isaac-lab scripts/queue/lm7p): what stays FIXED from each, and its own s1 levers
    "t21":  dict(std_floor=0.0912, guard_w=0.9392, track_err_w=-1.5077, yaw_err_w=-12.0),
    "t36":  dict(std_floor=0.1147, guard_w=0.0,    track_err_w=-0.5,    yaw_err_w=-12.0),
    "t37":  dict(std_floor=0.0794, guard_w=0.0,    track_err_w=-0.5,    yaw_err_w=-12.0),
    "best": dict(std_floor=0.0817, guard_w=0.0,    track_err_w=-1.4473, yaw_err_w=-12.0),
}
S2_PARENT = "best"       # = lm7p_enc_best_2026-11-13 @163650 (operator 2026-11-13 after the robot test); space_s2.yaml's center names it


def gene_err_split(cfg, vx_err_w: float, vy_err_w: float, yaw_err_w: float):
    """track_err (one weight on vx^2 + vy^2 + wz^2) -> track_vx, track_vy, and its yaw share folded into track_yaw."""
    R = cfg.rewards
    assert R.track_err.func is mdp.track_vel_err_l2 and R.track_yaw.func is mdp.track_yaw_err_l2, "[s2] the parent's track_err + track_yaw underneath"
    assert getattr(R, "track_vx", None) is None and getattr(R, "track_vy", None) is None, "[s2] no per-axis prices underneath"
    assert vx_err_w < 0 and vy_err_w < 0 and yaw_err_w < 0, "[s2] prices (negative)"
    R.track_err = None
    R.track_vx = RewTerm(func=mdp.track_vx_err_l2, weight=float(vx_err_w), params={"command_name": "base_velocity"})
    R.track_vy = RewTerm(func=mdp.track_vy_err_l2, weight=float(vy_err_w), params={"command_name": "base_velocity"})
    R.track_yaw.weight = float(yaw_err_w)


def gene_l1(cfg, lin_l1_w: float, yaw_l1_w: float):
    R = cfg.rewards
    assert getattr(R, "track_lin_l1", None) is None and getattr(R, "track_yaw_l1", None) is None, "[s2] no absolute-error prices underneath"
    if lin_l1_w < 0:
        R.track_lin_l1 = RewTerm(func=mdp.track_lin_err_l1, weight=float(lin_l1_w), params={"command_name": "base_velocity"})
    if yaw_l1_w < 0:
        R.track_yaw_l1 = RewTerm(func=mdp.track_yaw_err_l1, weight=float(yaw_l1_w), params={"command_name": "base_velocity"})


def gene_feet_yaw_w(cfg, feet_yaw_w: float):
    t = cfg.rewards.feet_yaw_bonus
    assert t.func is mdp.feet_yaw_bonus_turn_faded and t.weight == 3.0 and t.params["wz_ref"] == 0.5 and feet_yaw_w >= 0, "[s2] the parent's feet_yaw_bonus: turn-faded at 0.5 rad/s, weight 3"
    t.weight = float(feet_yaw_w)


def gene_torso_rel(cfg, torso_rel: float):
    """torso_stability_lin (torso_stability_bonus, std_lin .15, std_ang 1e6 = the linear half of the split) -> the command-relative
    twin at the same weight and std; only swapped at torso_rel > 0, so torso_rel 0 leaves the parent's term in place."""
    t = cfg.rewards.torso_stability_lin
    assert t.func is mdp.torso_stability_bonus and t.weight == 3.0 and t.params["std_lin"] == 0.15 and t.params["std_ang"] >= 1e5 \
        and 0.0 < torso_rel <= 1.0, "[s2] the parent's torso_stability_lin = the linear half (std_lin .15) at weight 3"
    cfg.rewards.torso_stability_lin = RewTerm(func=mdp.torso_stability_lin_rel, weight=t.weight,
                                              params={"asset_cfg": t.params["asset_cfg"], "std_lin": t.params["std_lin"],
                                                      "command_name": "base_velocity", "lin_rel": float(torso_rel)})


def study_s2(cfg, track_w=10.0, vx_err_w=None, vy_err_w=None, yaw_err_w=None, lin_l1_w=0.0, yaw_l1_w=0.0, feet_yaw_w=3.0, torso_rel=0.0,
             parent=None):
    """the parent's chain (study_s1 at its levers) + the band held at S2_BAND + the s2 levers. A lever left at None takes the parent's
    value (vx_err_w / vy_err_w = its track_err_w; yaw_err_w = its yaw_err_w + its track_err_w = track_err's yaw share folded in)."""
    P = S2_PARENTS[parent or S2_PARENT]
    study_s1(cfg, track_w=track_w, track_err_w=P["track_err_w"], std_floor=P["std_floor"], guard_w=P["guard_w"], yaw_err_w=P["yaw_err_w"])
    gene_enc(cfg, S2_BAND, SW_ENC01_JOINTS)
    assert getattr(cfg.curriculum, "enc_ladder", None) is None and cfg.events.encoder_offset.params["offset_range"] == (-S2_BAND, S2_BAND), "[s2] a held band, no ladder"
    gene_err_split(cfg,
                   P["track_err_w"] if vx_err_w is None else vx_err_w,
                   P["track_err_w"] if vy_err_w is None else vy_err_w,
                   (P["yaw_err_w"] + P["track_err_w"]) if yaw_err_w is None else yaw_err_w)
    if float(lin_l1_w) < 0.0 or float(yaw_l1_w) < 0.0:
        gene_l1(cfg, float(lin_l1_w), float(yaw_l1_w))
    if abs(float(feet_yaw_w) - 3.0) > 1e-9:
        gene_feet_yaw_w(cfg, feet_yaw_w)
    if float(torso_rel) > 0.0:
        gene_torso_rel(cfg, torso_rel)
