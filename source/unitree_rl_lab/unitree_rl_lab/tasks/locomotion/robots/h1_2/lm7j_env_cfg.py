"""lm7j (2026-11-02): arms, walking torso, legs in / out + the gradual stop, on lm7i_combo_soft.

Operator 2026-11-02 sim2sim (scripts/queue/lm7i/CONCLUSIONS.md): lm7i_combo_soft "Ok i like how this looks. Much calmer stance. Starts to
slowly do a split ... the legs seem to go inward when walking. The left leg. But i like this policy" -> "combo_soft seems to be the natural
parent ... Let's tackle the arm swinging and torso stability together + the leg inward / outward"; lm7i_arms "Arms still swing but a lot
less ... Looking at the shoulder line, it goes left and right while walking"; the combo "already answered the question - the reward weight is
too small for it". Accepted rows: arms6, arms10, torso15 ("Still worth trying"), stance, stopramp ("It goes in this batch"), combo.

Genes (scripts/queue/lm7j/HYPOTHESIS.md; parent ledger = the harvested lm7i_combo_soft log, last 100 it):
  arms6     joint_deviation_arms (L1 to default, 14 arm joints) -3 -> -6   (parent -3.07/s -> ~-6.15/s at the parent's behaviour)
  arms10    joint_deviation_arms -3 -> -10                                  (~-10.2/s)
  torso15   torso_stability_ang (3-axis torso ang vel, yaw relative to the command) std_ang .30 -> .15, weight 3 (parent +0.32/s = a mean
            kernel of 0.11 at std .30 -> ~0.45 rad/s typical; at .15 exp(-9) ~ 1e-4: FLAT where the robot lives, Bible rule 31 FLAG)
  stance    feet_too_far threshold_stand .45 -> .38; foot_stance_tracking 1.5 -> 3.0; feet_too_near threshold_walk .22 -> .24
  stopramp  the velocity command term -> mdp.SlewVelocityCommand (dec 1.0 / 1.0 m/s^2, 2.0 rad/s^2; acc instant): T = |v0| / a
  roll      + torso_roll_walk = mdp.torso_roll_rate_l2_walking (torso_link roll rate about its body x, squared, |cmd| >= .03) x ROLL_W
            (the parent's walking-gated mean wx^2 = 0.103 rad^2/s^2 -> -15 x 0.103 = -1.55/s, -6 % of the ~25.4/s return)
  ramp2     stopramp + an acceleration limit 0.5 / 0.5 m/s^2, 1.0 rad/s^2 (the slew's acc parameter)
  ffarms    gene A (robot 11-02, plant parity): the action term -> mdp.ArmGravityFFJointPositionActionCfg (same scale / offset / knee clip /
            joints): the generalized gravity torque of the 14 arm DOFs is written as the joint EFFORT target every physics step (the
            robot's AM_ARM_GRAV_FF, clip 25 Nm). No reward change; the arms stop sagging by g(q) / kp (checked: scripts/tools/check_arm_gravity_ff.py).
            The gene is THE PLANT, and the plant includes the bank: the 50 % policy-activation starts (lm7h gene_bank_sw, prob .5) come from
            BANK_SW_FF = v1 re-recorded with the ff ON (record_pose_bank.py --arm_ff, same milestone / recipe, 2026-11-02) so a bank start
            is p14h_stancewide's stand ON the compensated plant (arms on their targets, the lean the robot shows) instead of the sagged v1
            pose the ff would lift in the first ~0.1 s. The nine rows without gene A keep v1 (BANK_SW).
  clear     gene B (robot 11-02: the left toe scuff at 0.15 m/s; exam peak swing 1.6 / 2.3 cm vs the 8 cm target): feet_swing
            (lm7u.foot_clearance_swing_cmd) std .05 -> .001 (exp(-(z - .08)^2 / std), std UNSQUARED: sigma 22 cm -> 3.2 cm; 2 cm pays
            0.03 instead of 0.93, 5 cm 0.41, 8 cm 1.0), weight 3 -> 6; feet_clearance (foot_clearance_reward_cmd 9, the same flat
            kernel, maximal for a motionless loaded foot) weight 9 -> 0 (the RewardManager skips a weight-0 term; the key stays in env.yaml).
Rows: arms6 | arms10 | torso15 | stance | stopramp | combo (arms6 + torso15 + stance) | roll | ramp2 | combo_roll (arms6 + roll + stance) |
      ffarms (A) | clear (B) | combo_ff (combo + A) | combo_ff_clear (combo + A + B).
Every row = lm7i_env_cfg.combo_soft + its gene. roll, ramp2, combo_roll = the 2026-11-02 additions (operator-accepted); torso15 stays as the
kernel-width CONTROL (operator 11-02: "Control"). ffarms / clear / combo_ff / combo_ff_clear = the 2026-11-02 evening additions from the
robot tests (HYPOTHESIS.md C15-C18).

2026-11-03 rows on lm7j_combo @133393 (PARENT2; the Exam v3b re-land: every row but ffarms slides OUT after a walk, clear did not lift
the feet, the stop recoils; operator's genes 11-03):
  clear2    gene B re-formed: feet_swing std .001 -> .01 (sigma 3.2 -> 10 cm: a 2 cm swing pays exp(-(.06)^2 / .01) = 0.70 against the
            8 cm target, 5 cm 0.91 -- a gradient where the robot lives; at .001 it paid 0.03 and the policy stepped less instead),
            weight 6, feet_clearance 0, + a TARGET CURRICULUM: target_height 0.04 -> 0.08 m linear over RAMP_IT iterations starting
            RAMP_START_IT after the parent's iteration (mdp.reward_param_ramp on absolute ticks).
  stopramp2 gene_stopramp (dec 1.0 / 1.0 / 2.0, acc instant) + a curriculum on the deceleration: (1.0, 1.0, 2.0) -> (0.5, 0.5, 1.0)
            over the same ramp (mdp.slew_dec_ramp): 0.8 m/s stops in 0.8 s at the start, 1.6 s at the end.
  armsway   + arm_motion_walking (mdp.arm_motion_walking, the 8 upper-arm joint velocities squared while |cmd| > .05) x ARM_MOTION_W.
  legs      + feet_width_hold_standing (std .03, w0 = the anchor at the spawn / the last stop) x 3, feet_width_walking (.30, std .06)
            x 1.5, feet_yaw_hold_standing (std .1 rad) x 2. feet_too_near_split / feet_too_far_split / feet_crossed stay as they are.
Rows: stack (combo + ffarms + roll + clear2 + stopramp2 + armsway + legs) | stack_base (combo + ffarms + roll + clear2 + stopramp2) |
      legs (combo + legs). Every row = combo (bit-identical, the parent's own genes) + its 2026-11-03 genes; the warmstart is PARENT2.

Group D (2026-11-03 evening; robot 11-03: lm7j_clear / combo_ff_clear scuffed the LEFT toe twice at 0.15 m/s while sim2sim walks clean --
the exam apex is 2.2-2.6 cm and the real left foot carries ~1-4 cm of height-belief error; operator: "really make an emphasis on the
clearance; increase the weight if you need to", a stampeding leg accepted; + a floor-bump DR axis). Trunk = stack_base with clear2 REPLACED:
  clear8    feet_swing target_height 0.08 FIXED (no 4 -> 8 curriculum), std .01 (sigma 10 cm), weight 12 (2 cm pays 12 x 0.70 = 8.4 vs 12 at
            8 cm: the slope 2 -> 8 cm is 3.6 / step = 0.6 / cm, against the charges of a 6 cm higher swing (energy ~0.1-0.2, action_rate
            ~0.03, joint_vel_reg ~0.05, torso kernels <= 0.3: <= 0.6 / step in all) -> 12 holds, kept at the operator's number), feet_clearance 0.
  clear10   as clear8 with target 0.10 (aim 6-8 cm real after the sag): 2 cm pays 0.53 x 12 = 6.4, the slope 2 -> 10 cm 5.6 / step.
  bumps     the scene's flat plane -> a TerrainGenerator (curriculum OFF, seed 1): 10 x 20 patches of 8 x 8 m, 75 % MeshPlaneTerrainCfg (flat,
            z = 0) + 25 % HfRandomUniformTerrainCfg noise 0.5-2 cm (step 0.5 cm, heights drawn every 0.20 m and interpolated, mesh cells
            0.10 m), a 10 m flat border; the same physics material (friction 1.0 / 1.0, restitution 0). Sub-terrains are drawn per PATCH by
            proportion, env origins spread over the columns and random rows -> ~25 % of the envs on bumps, flat stays the default. Both
            reset paths (reset_root_state_uniform, reset_from_pose_bank) add env_origins (a bump patch's origin z = its centre height);
            the anchors are xy / yaw only; no height scan in the obs. mdp.bumps_proof (startup) prints the measured share.
Rows: stack_clear8 (D1) | stack_clear10 (D2) | stack_bumps (D3 = stack_base + bumps) | stack_clear8_bumps (D4 = D1 + bumps) |
      stack_full (E, operator 17:40 = row A with D4's clearance + bumps: ffarms + roll + stopramp2 + clear8 + armsway + legs + bumps).
"""
import math

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.commands.velocity_command import CornerVelocityCommand

from .lm7h_env_cfg import BANK_SW
from .lm7i_env_cfg import combo_soft
from .lm7u_env_cfg import _FEET

PARENT, IT = "lm7i_combo_soft_2026-11-01", 126394
BANK_SW_FF = "balance_p14h_stancewide_armdeploy_2026-10-29_ff_2026-11-02.pt"   # aspired-isaac-lab/assets/pose_banks/: BANK_SW re-recorded WITH the arm ff (gene A rows only)
PARENT2, IT2 = "lm7j_combo_2026-11-02", 133393   # the 2026-11-03 rows' parent (the lm7j_combo milestone, Exam v3b 61.8 / v3 64.1)
RAMP_START_IT, RAMP_IT = 300, 2000                # the curricula: flat through the 300-it critic warm-up, then 2000 it linear (48 000 ticks)
RAMP_START, RAMP_STEPS = (IT2 + RAMP_START_IT) * 24, RAMP_IT * 24   # absolute common_step_counter ticks: 3 208 632 -> 3 256 632 = it 133 693 -> 135 693
SWING_STD, SWING_T0, SWING_T1 = 0.01, 0.04, 0.08  # clear2: sigma 10 cm; the target ramps 4 -> 8 cm
DEC0, DEC1 = (1.0, 1.0, 2.0), (0.5, 0.5, 1.0)     # stopramp2: the deceleration limit (vx, vy m/s^2; wz rad/s^2) at the ramp's start / end
ARM_MOTION_W = -0.25                              # armsway: three upper-arm joints at 1 rad/s = 0.75 / step; one joint 0.25 / step
ROLL_W = -15.0   # lm7j_roll: the parent's walking-gated mean wx^2 = 0.103 rad^2/s^2 (90-it liveness, full-episode scale) -> -15 x 0.103 = -1.55 / s (HYPOTHESIS.md)


def _arms(cfg, w):
    d = cfg.rewards.joint_deviation_arms
    assert d.func is mdp.joint_deviation_l1 and d.weight == -3.0, "[arms] the parent's joint_deviation_arms -3.0 (L1, lm7i gene_arms)"
    assert getattr(cfg.rewards, "arm_swing", None) is None, "[arms] the parent has no arm_swing (lm7i gene_arms removed it)"
    d.weight = w


def gene_arms6(cfg): _arms(cfg, -6.0)
def gene_arms10(cfg): _arms(cfg, -10.0)


def gene_torso15(cfg):
    t = cfg.rewards.torso_stability_ang
    assert t.func is mdp.torso_stability_bonus and t.weight == 3.0, "[torso15] the parent's torso_stability_ang +3 (torso_stability_bonus)"
    assert t.params["std_ang"] == 0.3 and t.params["std_lin"] == 1e6 and t.params.get("command_name") == "base_velocity", \
        "[torso15] the parent's ang-only kernel std .30, yaw relative to base_velocity"
    t.params["std_ang"] = 0.15


def gene_stance(cfg):
    R = cfg.rewards
    far, near, st = R.feet_too_far, R.feet_too_near, R.foot_stance_tracking
    assert far.func.__name__ == "feet_too_far_split" and far.weight == -8.0 and far.params["threshold_stand"] == 0.45 and far.params["threshold_walk"] == 0.6, \
        "[stance] the parent's feet_too_far_split -8 (stand .45 / walk .60)"
    assert near.func.__name__ == "feet_too_near_split" and near.weight == -8.0 and near.params["threshold_stand"] == 0.28 and near.params["threshold_walk"] == 0.22, \
        "[stance] the parent's feet_too_near_split -8 (stand .28 / walk .22)"
    assert st.func.__name__ == "foot_stance_tracking_split" and st.weight == 1.5 and st.params["std"] == 0.08, \
        "[stance] the parent's foot_stance_tracking_split 1.5 @ std .08"
    far.params["threshold_stand"] = 0.38
    st.weight = 3.0
    near.params["threshold_walk"] = 0.24


def gene_stopramp(cfg, dec=(1.0, 1.0, 2.0), acc=(math.inf, math.inf, math.inf)):
    C = cfg.commands.base_velocity
    assert C.class_type is CornerVelocityCommand and C.heading_command and C.rel_heading_envs == 0.5 and C.rel_standing_envs == 0.25, \
        "[stopramp] the parent's CornerVelocityCommand (heading envs .5, standing envs .25)"
    cfg.commands.base_velocity = mdp.slew_command_cfg(C, dec=dec, acc=acc)


def gene_roll(cfg, w=None):
    R = cfg.rewards
    assert getattr(R, "torso_roll_walk", None) is None, "[roll] the parent has no walking roll-rate price"
    assert R.torso_ang_vel.func.__name__ == "body_ang_vel_l2_standing", "[roll] the standing side is torso_ang_vel (standing-gated)"
    R.torso_roll_walk = RewTerm(func=mdp.torso_roll_rate_l2_walking, weight=ROLL_W if w is None else w,
                                params={"command_name": "base_velocity", "cmd_thr": 0.03,
                                        "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])})


def gene_ffarms(cfg):
    """Gene A: the robot's arm gravity feed-forward in sim (mdp/lm7j.py ArmGravityFFJointPositionAction) + the pose bank recorded on that
    plant (BANK_SW_FF). A PLANT change, no reward change; one gene = "the ff plant", so the two never travel apart."""
    A = cfg.actions.JointPositionAction
    assert A.class_type is JointPositionAction and A.scale == 0.25 and A.use_default_offset and A.asset_name == "robot", \
        "[ffarms] the parent's plain JointPositionAction (scale .25, default offset)"
    assert A.clip == {".*_knee_joint": (0.45, 2.5)}, "[ffarms] the parent's knee clip rides along"
    cfg.actions.JointPositionAction = mdp.arm_gravity_ff_action_cfg(A)
    assert cfg.actions.JointPositionAction.arm_gravity_ff and cfg.actions.JointPositionAction.clip == A.clip
    B = cfg.events.reset_pose_bank
    assert B.func is mdp.reset_from_pose_bank_hold and B.params["bank"] == BANK_SW and B.params["prob"] == 0.5 and B.params["hold_s"] == 3.0, \
        "[ffarms] the parent's bank_sw (lm7h: v1 bank, prob .5, 3 s hold)"
    B.params["bank"] = BANK_SW_FF   # the same loader / prob / noise / hold: only the recorded plant changes


def gene_clear(cfg):
    """Gene B: a swing-clearance kernel that binds (feet_swing std .05 -> .001, 3 -> 6) and the flat feet_clearance switched off (9 -> 0)."""
    R = cfg.rewards
    c, sw = R.feet_clearance, R.feet_swing
    assert c.func is mdp.foot_clearance_reward_cmd and c.weight == 9.0 and c.params["target_height"] == 0.08 and c.params["std"] == 0.05 \
        and c.params["tanh_mult"] == 2.0, "[clear] the parent's feet_clearance 9 (foot_clearance_reward_cmd, .08 / std .05 / tanh 2)"
    assert sw.func is mdp.foot_clearance_swing_cmd and sw.weight == 3.0 and sw.params["target_height"] == 0.08 and sw.params["std"] == 0.05 \
        and sw.params["tanh_mult"] == 2.0 and sw.params["cmd_thr"] == 0.03, "[clear] the parent's feet_swing 3 (lm7u swing kernel, .08 / std .05 / tanh 2, thr .03)"
    c.weight = 0.0
    sw.params["std"] = 0.001
    sw.weight = 6.0


def arms6(cfg): combo_soft(cfg); gene_arms6(cfg)
def arms10(cfg): combo_soft(cfg); gene_arms10(cfg)
def torso15(cfg): combo_soft(cfg); gene_torso15(cfg)
def stance(cfg): combo_soft(cfg); gene_stance(cfg)
def stopramp(cfg): combo_soft(cfg); gene_stopramp(cfg)
def combo(cfg): combo_soft(cfg); gene_arms6(cfg); gene_torso15(cfg); gene_stance(cfg)
def roll(cfg): combo_soft(cfg); gene_roll(cfg)
def ramp2(cfg): combo_soft(cfg); gene_stopramp(cfg, acc=(0.5, 0.5, 1.0))
def combo_roll(cfg): combo_soft(cfg); gene_arms6(cfg); gene_roll(cfg); gene_stance(cfg)   # the three focus items with the roll-rate price (torso15 = the kernel-width CONTROL)
def ffarms(cfg): combo_soft(cfg); gene_ffarms(cfg)                 # gene A alone (the plant-parity single)
def clear(cfg): combo_soft(cfg); gene_clear(cfg)                   # gene B alone
def combo_ff(cfg): combo(cfg); gene_ffarms(cfg)                    # the combo row's gene set, bit-identical, + A
def combo_ff_clear(cfg): combo(cfg); gene_ffarms(cfg); gene_clear(cfg)   # combo + A + B


# ── 2026-11-03 genes (on lm7j_combo @133393) ──
def gene_clear2(cfg):
    """clear' = gene B with a sigma that has a gradient at a 2 cm swing (std .01) and the target ramped 4 -> 8 cm (mdp.reward_param_ramp)."""
    R = cfg.rewards
    c, sw = R.feet_clearance, R.feet_swing
    assert c.func is mdp.foot_clearance_reward_cmd and c.weight == 9.0 and c.params["target_height"] == 0.08 and c.params["std"] == 0.05 \
        and c.params["tanh_mult"] == 2.0, "[clear2] the parent's feet_clearance 9 (foot_clearance_reward_cmd, .08 / std .05 / tanh 2)"
    assert sw.func is mdp.foot_clearance_swing_cmd and sw.weight == 3.0 and sw.params["target_height"] == 0.08 and sw.params["std"] == 0.05 \
        and sw.params["tanh_mult"] == 2.0 and sw.params["cmd_thr"] == 0.03, "[clear2] the parent's feet_swing 3 (lm7u swing kernel, .08 / std .05 / tanh 2, thr .03)"
    assert getattr(cfg.curriculum, "swing_target", None) is None, "[clear2] no swing-target curriculum underneath"
    c.weight = 0.0
    sw.params["std"] = SWING_STD
    sw.params["target_height"] = SWING_T0   # the ramp's start value (the curriculum writes the same at its first call)
    sw.weight = 6.0
    cfg.curriculum.swing_target = CurrTerm(func=mdp.reward_param_ramp,
                                           params={"reward_term_name": "feet_swing", "param": "target_height", "value_from": SWING_T0, "value_to": SWING_T1,
                                                   "start_step": RAMP_START, "ramp_steps": RAMP_STEPS})


def gene_stopramp2(cfg):
    """stopramp' = the slew command (dec 1.0 / 1.0 / 2.0, acc instant) + the deceleration ramped to half (mdp.slew_dec_ramp)."""
    gene_stopramp(cfg, dec=DEC0)
    assert getattr(cfg.curriculum, "slew_dec", None) is None, "[stopramp2] no slew-dec curriculum underneath"
    cfg.curriculum.slew_dec = CurrTerm(func=mdp.slew_dec_ramp,
                                       params={"command_name": "base_velocity", "dec_from": DEC0, "dec_to": DEC1, "start_step": RAMP_START, "ramp_steps": RAMP_STEPS})


def gene_armsway(cfg):
    R = cfg.rewards
    assert getattr(R, "arm_motion_walking", None) is None, "[armsway] no arm-motion price underneath"
    assert R.joint_vel_reg.weight == -0.01 and R.joint_deviation_arms.weight == -6.0, "[armsway] the parent's joint_vel_reg -0.01 / joint_deviation_arms -6 (combo)"
    R.arm_motion_walking = RewTerm(func=mdp.arm_motion_walking, weight=ARM_MOTION_W,
                                   params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(mdp.ARM_MOTION_JOINTS)), "command_name": "base_velocity", "cmd_thr": 0.05})


def gene_legs(cfg):
    R = cfg.rewards
    for n in ("feet_width_hold_standing", "feet_width_walking", "feet_yaw_hold_standing"):
        assert getattr(R, n, None) is None, f"[legs] no {n} underneath"
    assert R.feet_too_near.func.__name__ == "feet_too_near_split" and R.feet_too_near.params["threshold_stand"] == 0.28 and R.feet_too_near.params["threshold_walk"] == 0.24 \
        and R.feet_too_far.func.__name__ == "feet_too_far_split" and R.feet_too_far.params["threshold_stand"] == 0.38 and R.feet_crossed.weight == -8.0, \
        "[legs] the walls stay: feet_too_near .28 / .24, feet_too_far .38 / .60, feet_crossed -8 (combo's stance gene)"
    E = cfg.events
    assert E.reanchor_on_stop.func is mdp.reanchor_on_stop and E.capture_spawn_state.func is mdp.capture_spawn_state, "[legs] the anchor events (the feet anchor rides on them)"
    R.feet_width_hold_standing = RewTerm(func=mdp.feet_width_hold_standing, weight=3.0, params={"asset_cfg": _FEET, "std": 0.03, "command_name": "base_velocity", "cmd_thr": 0.03})
    R.feet_width_walking = RewTerm(func=mdp.feet_width_walking, weight=1.5, params={"asset_cfg": _FEET, "target": 0.30, "std": 0.06, "command_name": "base_velocity", "cmd_thr": 0.03})
    R.feet_yaw_hold_standing = RewTerm(func=mdp.feet_yaw_hold_standing, weight=2.0, params={"asset_cfg": _FEET, "std": 0.1, "command_name": "base_velocity", "cmd_thr": 0.03})


def stack_base(cfg): combo(cfg); gene_ffarms(cfg); gene_roll(cfg); gene_clear2(cfg); gene_stopramp2(cfg)   # B: combo + 1 + 2 + 3 + 4
def stack(cfg): stack_base(cfg); gene_armsway(cfg); gene_legs(cfg)                                          # A: + 5 + 6
def legs(cfg): combo(cfg); gene_legs(cfg)                                                                   # C: combo + 6


# ── group D genes ──
BUMP_NOISE, BUMP_STEP, BUMP_PROP, BUMP_SAMPLE_M = (0.005, 0.02), 0.005, 0.25, 0.2   # the exam's stumble field: caps 0.4-2 cm on R 0.15 m spheres (unitree_mujoco stumble_field.py)


def gene_clear_fixed(cfg, target, weight=12.0):
    """clear8 / clear10: gene B's two kernels re-sized for EMPHASIS -- feet_swing std .01 (sigma 10 cm), a FIXED target, weight 12;
    feet_clearance 0. No curriculum (the operator wants the margin from the first iteration)."""
    R = cfg.rewards
    c, sw = R.feet_clearance, R.feet_swing
    assert c.func is mdp.foot_clearance_reward_cmd and c.weight == 9.0 and c.params["target_height"] == 0.08 and c.params["std"] == 0.05, \
        "[clear_fixed] the parent's feet_clearance 9 (foot_clearance_reward_cmd, .08 / std .05)"
    assert sw.func is mdp.foot_clearance_swing_cmd and sw.weight == 3.0 and sw.params["target_height"] == 0.08 and sw.params["std"] == 0.05 \
        and sw.params["cmd_thr"] == 0.03, "[clear_fixed] the parent's feet_swing 3 (lm7u swing kernel, .08 / std .05, thr .03)"
    assert getattr(cfg.curriculum, "swing_target", None) is None, "[clear_fixed] no swing-target curriculum (clear2 is REPLACED, not stacked)"
    c.weight = 0.0
    sw.params["std"] = SWING_STD
    sw.params["target_height"] = float(target)
    sw.weight = float(weight)


def gene_bumps(cfg):
    """The floor DR: 25 % of the envs on 0.5-2 cm random bumps, 75 % flat, same material; curriculum OFF; a startup proof event."""
    T = cfg.scene.terrain
    assert T.terrain_type == "plane" and T.terrain_generator is None and T.physics_material.static_friction == 1.0 \
        and T.physics_material.dynamic_friction == 1.0 and T.physics_material.restitution == 0.0 and T.collision_group == -1, \
        "[bumps] the parent's flat plane (friction 1.0 / 1.0, restitution 0)"
    assert getattr(cfg.events, "bumps_proof", None) is None and getattr(cfg.curriculum, "terrain_levels", None) is None, "[bumps] no terrain curriculum / proof underneath"
    gen = TerrainGeneratorCfg(
        seed=1, curriculum=False, use_cache=False, size=(8.0, 8.0), border_width=10.0, num_rows=10, num_cols=20,
        horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75, color_scheme="none",
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0 - BUMP_PROP),
            "bumps": terrain_gen.HfRandomUniformTerrainCfg(proportion=BUMP_PROP, noise_range=BUMP_NOISE, noise_step=BUMP_STEP, downsampled_scale=BUMP_SAMPLE_M),
        })
    cfg.scene.terrain = TerrainImporterCfg(prim_path=T.prim_path, terrain_type="generator", terrain_generator=gen, collision_group=T.collision_group,
                                           physics_material=T.physics_material, max_init_terrain_level=None, debug_vis=False)
    cfg.events.bumps_proof = EventTerm(func=mdp.bumps_proof, mode="startup")
    print(f"[lm7j] bumps terrain cfg: {gen.num_rows} x {gen.num_cols} patches of {gen.size} m, flat {1.0 - BUMP_PROP:.2f} / bumps {BUMP_PROP:.2f} "
          f"(noise {BUMP_NOISE} m, step {BUMP_STEP}, sampled every {BUMP_SAMPLE_M} m, cells {gen.horizontal_scale} m), border {gen.border_width} m, curriculum off, seed {gen.seed}", flush=True)


def _stack_base_noclear(cfg): combo(cfg); gene_ffarms(cfg); gene_roll(cfg); gene_stopramp2(cfg)   # trunk B without clear2 (replaced by a D clearance gene)
def stack_clear8(cfg): _stack_base_noclear(cfg); gene_clear_fixed(cfg, 0.08, 12.0)                # D1
def stack_clear10(cfg): _stack_base_noclear(cfg); gene_clear_fixed(cfg, 0.10, 12.0)               # D2
def stack_bumps(cfg): stack_base(cfg); gene_bumps(cfg)                                              # D3 = B + the floor DR
def stack_clear8_bumps(cfg): stack_clear8(cfg); gene_bumps(cfg)                                     # D4 = D1 + the floor DR (the deploy candidate)
def stack_full(cfg): stack_clear8(cfg); gene_armsway(cfg); gene_legs(cfg); gene_bumps(cfg)                      # E = row A with D4's clearance + bumps
