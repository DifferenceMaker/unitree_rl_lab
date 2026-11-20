"""study s5_stillness (2026-11-20): torso stillness + the arm hits to 0, on lm7r_encup_t28w @178146 -- the study gene.

Operator 2026-11-20: "torso stability really will be our next study 100%" (on hardware the robot leaned back; the start of walking was
the hardest and he had to hold it); arm hits: "I lean more towards 0"; no encoder ladder in s5 (the band held at +-0.015).
Parent = milestone lm7r_encup_t28w_2026-11-20 @178146 = lm7q_enc_up + t28's nine s4 weights, trained with the encoder-offset ladder
+-0.010 -> 0.012 -> 0.015 rad (0.015 from ~+850 of its 3500 it). Exam #3 rev 3 63.1 (stability 19.5, gait 5.4, homing 3.0), arm hits
23 (every one torso_link x shoulder_roll_link = the upper arm rolling inward, 0 while walking), 12 lifts, walk_stand sway 0.023 m/s.

Chain = lm7r_env_cfg.lm7r_encup_t28w WITHOUT its ladder: study_s4(**S4_T28) (t28's nine weights, the band held at +-0.010 = S2_BAND
underneath) -> gene_band_hold re-sets that band to +-0.015 and HOLDS it (no curriculum term: the ladder's rung lives on the env object,
not in the checkpoint, so a ladder would restart at +-0.010; held, every s5 trial trains at the parent's final band from iteration 0).
So the CENTRE's env.yaml differs from the parent milestone's in exactly two places, both the band:
    events.encoder_offset.params.offset_range   (-0.01, 0.01) -> (-0.015, 0.015)
    curriculum.enc_ladder                       present (rungs 0.010 / 0.012 / 0.015) -> absent
A NEW term at weight 0 is not added at all (bit-identical: the centre is the parent's env).

Levers = <reward term>_w (operator 2026-11-20 ~18:45: "I like it when the levers are as close to the terms as possible"); centre = the
parent; "pays" = the parent's Episode_Reward per second, mean of its last 100 iterations (lm7r_encup_t28w.log):
  track_lin_vel_xy_w       track_lin_vel_xy (study_s4's track_w), pays +4.76 /s                                   parent 10.5824
  base_angular_velocity_w  ang_vel_xy_l2 on the pelvis (roll + pitch rate^2), ungated, -0.033                    parent -0.3
  flat_orientation_l2_w    |g_xy|^2 of the pelvis' projected gravity, ungated, -0.014                            parent -4.5
  torso_roll_walk_w        torso roll rate^2 while |cmd| >= 0.03 (lm7j), -0.504                                   parent -15.0
  torso_ang_vel_w          torso |omega|^2 while STANDING (body_ang_vel_l2_standing), -0.103                      parent -2.0
  torso_pitch_w            NEW mdp.torso_pitch_bonus exp(-|theta| / 3 deg) (laplace), INCOME, ungated             parent 0 (absent)
  arm_adduct_w             NEW mdp.arm_adduct_hinge sum_{L,R} max(0, q_in - 0.06 rad), HINGE, ungated             parent 0 (absent)
  joint_deviation_arms_w   sum |q - q_default| over the 14 arm DOFs, ungated (study_s4's arm_dev_w), -4.91         parent -8.5399
  arm_motion_walking_w     sum qd^2 over the 8 upper-arm DOFs while |cmd| > 0.05 (study_s4's arm_motion_w), -3.00  parent -0.8474
  (operator 2026-11-20 ~18:25: "I would much rather have frozen arms" -- frozen > front-back pitch swing > sideways, never inward: the
  two s4 arm prices, grip -0.72 each in s3 / s4, join as levers.)
  arm_side_sway_w          mdp.arm_side_sway_walking (roll + yaw qd^2 walking) is NOT a lever (operator ~18:40 "Won't this be
                           overengineering the problem?" -- arm_motion_walking already prices every upper-arm axis); kept at 0 = absent.
Everything else FIXED at the parent's values: t28's other six s4 weights, the pushes (push_velocity / sustained_push curricula, resumed
at the top 1.5 m/s / 50 N by --resume_curriculum), the band +-0.015.
"""
import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.locomotion import mdp

from .lm7l_env_cfg import SW_ENC01_JOINTS
from .lm7r_env_cfg import S4_T28
from .study_env_cfg import study_s4, S2_BAND

S5_PARENT_MILESTONE, S5_PARENT_IT = "lm7r_encup_t28w_2026-11-20", 178146
S5_BAND = 0.015            # rad on the 13 leg + torso joints: the parent's final rung, held (operator: no ladder in s5)
S5_PARENT_WEIGHTS = {      # lever -> (reward term, the parent's weight = the lever's default); all four exist in the parent's env.yaml
    "base_angular_velocity_w": ("base_angular_velocity", -0.3),
    "flat_orientation_l2_w":   ("flat_orientation_l2",   -4.5),
    "torso_roll_walk_w":       ("torso_roll_walk",       -15.0),
    "torso_ang_vel_w":         ("torso_ang_vel",         -2.0),
}
# torso_pitch's kernel width (Rule 31): sized from the parent's measured torso pitch in Isaac -- see the s5 HYPOTHESIS (aspired-isaac-lab
# scripts/queue/s5/HYPOTHESIS.md) for the distribution, the earned fraction and the slope table.
# Shape + width (2026-11-20, aspired-isaac-lab scripts/queue/s5/probe_s5.py --hist_out, 2048 envs x 2400 steps = 4.9 M env-steps per kind;
# the parent's pitch: walking +1.35 +- 2.73 deg, standing -0.07 +- 2.34, mean |theta| 1.94 (the mean action; training rollouts 2.02)):
#   gauss   sigma 1.16 earns 37 % (Rule 31) but its slope is 0 at upright and 0.7 % / 0 / 0 of its peak at 3 / 5 / 8 deg;
#   laplace sigma 1.19 earns 37 %, slope 100 / 19 / 8 / 1.5 / 0.1 % of its peak at 0 / 2 / 3 / 5 / 8 deg;
#   laplace sigma 3.0  earns 61 % (det 0.609, train 0.603), slope 100 / 51 / 37 / 19 / 6.9 % at 0 / 2 / 3 / 5 / 8 deg.
# Chosen: laplace sigma 3.0. The Laplace slope at the parent's typical |theta| (~2 deg) is (1/sigma) exp(-2/sigma): best at sigma 2
# (0.184 /deg), only 7 % lower at sigma 3 (0.171), so the gradient stays where the robot lives (Rule 31's intent) while the 5-8 deg start
# lean the operator held keeps a pull (0.063 / 0.023 /deg per unit weight; the Gaussian at 1.16: 0). The higher earned share is the price.
TORSO_PITCH_SHAPE = "laplace"    # operator 2026-11-20 ~18:25: "doesn't necessarily have to be a kernel. It can be what works the best"
TORSO_PITCH_SIGMA_DEG = 3.0
TORSO_PITCH_THETA0_DEG = 0.0     # the kernel's centre (upright); the knob for the Isaac (+1.4 walking) vs MuJoCo (-1.6) pitch gap, not moved
TORSO_PITCH_CMD_THR = 0.03       # the proof line's walking / standing split only (the reward is ungated)
# arm_adduct's thresholds (rad, measured INWARD = toward the torso; MEASURED 2026-11-20 in the exam's MuJoCo model, scripts
# mj_shoulder_torso_onset.py / mesh_vs_hull_onset.py): the shoulder_roll_link convex hull touches the torso mesh at 0.102-0.108 rad inward
# for shoulder pitch -0.1..1.2 (0.137-0.169 at pitch -0.4..-0.6, 0.124-0.155 at 1.4-1.57; only shoulder pitch matters for this pair);
# q_safe = 0.06 = onset - 0.04 rad of margin (the +-0.01 encoder band + push overshoot). The exam's hits are the HULL (927 cm3 vs the real
# mesh's 381 cm3; real meshes would first touch near 0.33-0.35 rad), and Isaac's torso collider (one 0.08 x 0.16 x 0.10 m box at z 0.10-0.20)
# stays >= 116 mm from shoulder_roll_link over the whole joint grid, so training can never see this contact -- the hinge is the only
# signal; the operator dislikes inward shoulder roll as a behaviour regardless. Deploy default roll = 0.0 both arms (q_in = 0: pays 0).
ARM_ADDUCT_Q_SAFE = {"left": 0.06, "right": 0.06}
ARM_ADDUCT_Q_SAFE_MEASURED = True       # False = a placeholder: the gene then refuses it unless S5_ALLOW_QSAFE_PLACEHOLDER=1 (smokes only)
ARM_SIDE_SWAY_CMD_THR = 0.05     # = arm_motion_walking's gate (|cmd| > 0.05)


def gene_band_hold(cfg, band: float = S5_BAND):
    """the encoder-offset band study_s4 holds at +-S2_BAND, re-set to +-band and held (no curriculum term). Loud when the event underneath
    is not study_s4's (one band, a reset event, the 13 joints, the policy in the encoder frame) or when a ladder is present."""
    ev = getattr(cfg.events, "encoder_offset", None)
    assert ev is not None and ev.func is mdp.randomize_encoder_offset and ev.mode == "reset" and "offset_range_per_joint" not in ev.params \
        and tuple(ev.params["offset_range"]) == (-S2_BAND, S2_BAND), f"[s5] study_s4's band +-{S2_BAND} (one band, reset event) underneath"
    assert list(ev.params["asset_cfg"].joint_names) == list(SW_ENC01_JOINTS), "[s5] the 13 leg + torso joints underneath"
    assert cfg.observations.policy.joint_pos_rel.func is mdp.joint_pos_rel_encoder, "[s5] the policy reads the encoder frame"
    assert getattr(cfg.curriculum, "enc_ladder", None) is None, "[s5] no ladder: the band is HELD"
    assert band > S2_BAND, f"[s5] the held band is wider than study_s4's {S2_BAND}"
    ev.params["offset_range"] = (-float(band), float(band))


def gene_parent_weight(cfg, lever: str, weight: float):
    """one s5 lever on a term the parent carries = its WEIGHT (S5_PARENT_WEIGHTS). Loud on a missing term (a dead lever), on a term not
    at the parent's weight underneath, and on a weight that is not negative (all four are prices; 0 would drop the term)."""
    term, w0 = S5_PARENT_WEIGHTS[lever]
    t = getattr(cfg.rewards, term, None)
    assert t is not None, f"[s5] rewards.{term} is missing from the cfg: {lever} would be a dead lever"
    assert t.weight == w0, f"[s5] the parent's {term} = {w0} underneath, got {t.weight}"
    assert float(weight) < 0.0, f"[s5] {lever} is a price (negative), got {weight}"
    t.weight = float(weight)


def gene_torso_pitch(cfg, weight: float, sigma_deg: float = TORSO_PITCH_SIGMA_DEG, shape: str = TORSO_PITCH_SHAPE):
    """+ rewards.torso_pitch = mdp.torso_pitch_bonus at `weight` (> 0, an income), sigma in rad in the cfg"""
    assert getattr(cfg.rewards, "torso_pitch", None) is None, "[s5] no torso_pitch term underneath"
    assert float(weight) > 0.0 and float(sigma_deg) > 0.0, f"[s5] torso_pitch is an income (w > 0), got w {weight}, sigma {sigma_deg} deg"
    cfg.rewards.torso_pitch = RewTerm(func=mdp.torso_pitch_bonus, weight=float(weight),
                                      params={"sigma": math.radians(float(sigma_deg)), "theta0": math.radians(TORSO_PITCH_THETA0_DEG), "shape": shape,
                                              "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                                              "command_name": "base_velocity", "cmd_thr": TORSO_PITCH_CMD_THR})


def gene_arm_adduct(cfg, weight: float, q_safe: dict = None):
    """+ rewards.arm_adduct = mdp.arm_adduct_hinge at `weight` (< 0, a price) with the per-side inward thresholds (rad)"""
    q = dict(q_safe or ARM_ADDUCT_Q_SAFE)
    if q_safe is None and not ARM_ADDUCT_Q_SAFE_MEASURED:
        assert os.environ.get("S5_ALLOW_QSAFE_PLACEHOLDER") == "1", \
            "[s5] arm_adduct q_safe is still the PLACEHOLDER (the MuJoCo contact-onset number has not been entered): refused outside a smoke"
        print(f"[s5] !!! arm_adduct q_safe = PLACEHOLDER {q} (S5_ALLOW_QSAFE_PLACEHOLDER=1: a smoke, never a trial)", flush=True)
    assert getattr(cfg.rewards, "arm_adduct", None) is None, "[s5] no arm_adduct term underneath"
    assert float(weight) < 0.0, f"[s5] arm_adduct is a price (negative), got {weight}"
    assert set(q) == {"left", "right"} and all(-0.38 < float(v) < 0.38 for v in q.values()), f"[s5] q_safe per side inside the roll limit 0.38, got {q}"
    cfg.rewards.arm_adduct = RewTerm(func=mdp.arm_adduct_hinge, weight=float(weight),
                                     params={"q_safe_left": float(q["left"]), "q_safe_right": float(q["right"]), "asset_cfg": SceneEntityCfg("robot"),
                                             "command_name": "base_velocity", "cmd_thr": TORSO_PITCH_CMD_THR})


def gene_arm_side_sway(cfg, weight: float):
    """+ rewards.arm_side_sway = mdp.arm_side_sway_walking at `weight` (< 0) on both shoulder roll + yaw, gated like arm_motion_walking"""
    am = cfg.rewards.arm_motion_walking
    assert am.func is mdp.arm_motion_walking and am.weight < 0 and am.params["cmd_thr"] == ARM_SIDE_SWAY_CMD_THR, \
        "[s5] arm_motion_walking (a live price, |cmd| > 0.05) underneath: the side price sits ON TOP of it"
    assert getattr(cfg.rewards, "arm_side_sway", None) is None, "[s5] no arm_side_sway term underneath"
    assert float(weight) < 0.0, f"[s5] arm_side_sway is a price (negative), got {weight}"
    cfg.rewards.arm_side_sway = RewTerm(func=mdp.arm_side_sway_walking, weight=float(weight),
                                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(mdp.S5_SIDE_JOINTS)),
                                                "command_name": "base_velocity", "cmd_thr": ARM_SIDE_SWAY_CMD_THR})


def study_s5(cfg, track_lin_vel_xy_w=10.5824, base_angular_velocity_w=-0.3, flat_orientation_l2_w=-4.5, torso_roll_walk_w=-15.0,
             torso_ang_vel_w=-2.0, torso_pitch_w=0.0, arm_adduct_w=0.0, joint_deviation_arms_w=-8.5399, arm_motion_walking_w=-0.8474,
             arm_side_sway_w=0.0):
    """the parent's chain (study_s4 at t28's nine weights; track_lin_vel_xy_w / joint_deviation_arms_w / arm_motion_walking_w passed
    through as study_s4's own track_w / arm_dev_w / arm_motion_w, whose genes assert t57's weights underneath and that the value has the
    right sign) + the band held at +-0.015 + the s5 levers. All defaults = the parent: the centre's env.yaml == the milestone's apart from
    the band (see the module docstring). A new term's lever at 0 = the term absent; its sign is checked by its gene (torso_pitch_w > 0
    income, arm_adduct_w < 0 price). arm_side_sway_w is not a lever of the s5 space (0 = absent)."""
    study_s4(cfg, **{**S4_T28, "track_w": float(track_lin_vel_xy_w), "arm_dev_w": float(joint_deviation_arms_w),
                     "arm_motion_w": float(arm_motion_walking_w)})
    gene_band_hold(cfg, S5_BAND)
    for lever, w in (("base_angular_velocity_w", base_angular_velocity_w), ("flat_orientation_l2_w", flat_orientation_l2_w),
                     ("torso_roll_walk_w", torso_roll_walk_w), ("torso_ang_vel_w", torso_ang_vel_w)):
        gene_parent_weight(cfg, lever, w)
    if float(torso_pitch_w) != 0.0:
        gene_torso_pitch(cfg, torso_pitch_w)
    if float(arm_adduct_w) != 0.0:
        gene_arm_adduct(cfg, arm_adduct_w)
    if float(arm_side_sway_w) != 0.0:
        gene_arm_side_sway(cfg, arm_side_sway_w)
