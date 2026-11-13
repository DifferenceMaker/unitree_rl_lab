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
# study to find what we did with s1 but now with vy and wz too" / "we shouldn't be too cautious about the hyperparameter limits".
# s2 = the PARENT's cfg (an lm7p row: study_s1 levers + the encoder band) with
#   * the band HELD at +-0.010 rad in every trial (no ladder: the parent already walks at the top rung; the exam cannot be allowed to
#     trade the band away, so it is not a lever),
#   * three s1 levers that ended ON a bound of the pilot's range, searched again in a wider range: track_w, track_err_w, yaw_err_w,
#   * std_floor and guard_w FIXED at the parent's values (std_floor had a flat interior optimum, the guard only cost),
#   * three new levers for the axes s1 could not move (walks_vy >= 0.9 in 2 / 36 trials, walks_wz in 0 / 36):
#       vy_err_w   + track_vy = mdp.track_vy_err_l2, the lateral L2 price (0 = absent)
#       spin_prob  the command sampler's share of pure in-place turns (vx = vy = 0, |wz| >= 0.2): cfg default 0
#       slow_prob  its share of small linear commands, vx / vy redrawn inside +-S2_SLOW_BAND (wz kept): cfg default 0
# The parent is plugged in by S2_PARENT (its fixed values) + the space file's center (its milestone and lever values).
from .lm7l_env_cfg import SW_ENC01_JOINTS, gene_enc   # noqa: E402

S2_BAND = 0.010          # rad: p16c_sw_enc01's band = the ladder's top rung, held
S2_SLOW_BAND = 0.15      # m/s: slow_prob redraws vx / vy inside +-0.15 = the exam's weak zone (the 0.05-0.15 rungs)
S2_PARENTS = {           # the lm7p rows (aspired-isaac-lab scripts/queue/lm7p): what stays FIXED from each
    "t21":  dict(std_floor=0.0912, guard_w=0.9392),
    "t36":  dict(std_floor=0.1147, guard_w=0.0),
    "t37":  dict(std_floor=0.0794, guard_w=0.0),
    "best": dict(std_floor=0.0817, guard_w=0.0),
}
S2_PARENT = "best"       # PLACEHOLDER until the lm7p rows land (operator picks); space_s2.yaml's center must name the same row


def gene_vy_err(cfg, vy_err_w: float):
    R = cfg.rewards
    assert getattr(R, "track_vy", None) is None and vy_err_w < 0, "[s2] no track_vy underneath; a price (negative); 0 = absent"
    R.track_vy = RewTerm(func=mdp.track_vy_err_l2, weight=float(vy_err_w), params={"command_name": "base_velocity"})


def gene_cmd_mix(cfg, spin_prob: float, slow_prob: float):
    """the command sampler's buckets (mdp/commands/velocity_command.py: one draw, exclusive: corner, then slow, then spin)."""
    c = cfg.commands.base_velocity
    assert (c.spin_prob or 0.0) == 0.0 and (c.slow_prob or 0.0) == 0.0 and c.spin_min == 0.2, "[s2] the parent samples no spin / slow commands"
    assert 0.0 <= spin_prob and 0.0 <= slow_prob and (c.corner_prob or 0.0) + spin_prob + slow_prob <= 1.0, "[s2] the buckets fit in one draw"
    c.spin_prob = float(spin_prob)
    c.slow_prob = float(slow_prob)
    if slow_prob > 0.0:
        c.slow_band = S2_SLOW_BAND


def study_s2(cfg, track_w=10.0, track_err_w=-1.4473, yaw_err_w=-12.0, vy_err_w=0.0, spin_prob=0.0, slow_prob=0.0, parent=None):
    """the parent's chain (study_s1 at its levers, the fixed ones from S2_PARENTS) + the band held at S2_BAND + the s2 levers. At the
    space's center (the parent's own track_w / track_err_w / yaw_err_w, the three new levers at 0) the cfg equals the parent
    milestone's except for the band: a fixed +-0.010 event instead of the ladder that climbed to it."""
    fixed = S2_PARENTS[parent or S2_PARENT]
    study_s1(cfg, track_w=track_w, track_err_w=track_err_w, std_floor=fixed["std_floor"], guard_w=fixed["guard_w"], yaw_err_w=yaw_err_w)
    gene_enc(cfg, S2_BAND, SW_ENC01_JOINTS)
    assert getattr(cfg.curriculum, "enc_ladder", None) is None and cfg.events.encoder_offset.params["offset_range"] == (-S2_BAND, S2_BAND), "[s2] a held band, no ladder"
    if float(vy_err_w) < 0.0:
        gene_vy_err(cfg, vy_err_w)
    if float(spin_prob) > 0.0 or float(slow_prob) > 0.0:
        gene_cmd_mix(cfg, spin_prob, slow_prob)
