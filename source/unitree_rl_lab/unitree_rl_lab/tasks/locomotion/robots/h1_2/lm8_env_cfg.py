"""LM8 (2026-10-22): recipe transplants, trained from SCRATCH on our body -- the published walking ledgers instead of ours.

Why scratch: every row changes the observation (a gait clock and/or a frame history), which no warmstart can load. Why these five
(notes/walking_recipes_vs_lm7_2026-10-21.md): #83 is the one deployed recipe that handles standing inside the gait term; d_inverse is
our own ledger re-weighted "walking-first, standing as a corner case"; unitree_rl_gym is the robot's own published walk; Menlo is the
no-clock camp done with a privileged critic and targeted DR; humanoid-gym is the reference-pose family.

What is OURS in every row: the scene, body (comx06 hand790), actions (27 joints, the walk clip), the walk command box + the level
curricula (the ladder principle for scratch), the push events and the observation noise. What is THEIRS: rewards and weights (as
published, at their numeric scale), the clock, the history, the standing share, the episode length, their DR where published.
Weights of legged_gym / humanoid-gym recipes are per their own dt (0.005 x 4 / 0.001 x 10): Isaac multiplies ours by 0.02, so absolute
incomes differ by a constant; ratios are preserved, which is what matters under an adaptive learning rate.
Runner: LM8PPORunnerCfg (GuardedPPO, no mirror loss: the mirror map is single-frame per term and cannot flatten a history).
"""
import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import ARM_JOINT_REGEX, BODY_JOINT_REGEX, RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .lm7e_env_cfg import MENLO_STD, _FEET_SENSOR, gene_dr_all, gene_pc_contacts
from .lm7u_env_cfg import _FEET, _lm7u_stack, gene_lm7d_base

_ROBOT = SceneEntityCfg("robot")
_HIPS = SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])
_TORSO = SceneEntityCfg("robot", joint_names=["torso_joint"])
_ARMS = SceneEntityCfg("robot", joint_names=ARM_JOINT_REGEX)
_BODY = SceneEntityCfg("robot", joint_names=BODY_JOINT_REGEX, preserve_order=True)
_LEGS_TORSO = SceneEntityCfg("robot", joint_names=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "torso_joint"])
_SWING_LEGS = SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
                                                   "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint"], preserve_order=True)
_NON_FEET = SceneEntityCfg("contact_forces", body_names=["(?!.*ankle_roll).*"])
STD_WIDE = math.sqrt(0.25)      # the legged_gym / #83 / G1 tracking sigma: exp(-e^2 / 0.25)
STD_HGYM = math.sqrt(0.2)       # humanoid-gym: exp(-5 e^2)
H1_2_HEIGHT = 0.98              # the trunk's base_height target (G1 recipes say 0.77-0.89 for their robot)


def _track(w_lin, w_ang, std):
    return (RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=w_lin, params={"command_name": "base_velocity", "std": std}),
            RewTerm(func=mdp.track_ang_vel_z_exp, weight=w_ang, params={"command_name": "base_velocity", "std": std}))


def _common(cfg, *, episode_s, standing, clock=None, history=0, tilt=0.8):
    """What every transplant does to the walk base: episode length, standing share, tilt termination, the visible clock, the actor's
    frame history; the tracking ladder goes (each recipe fixes its sigma), the level curricula stay (they key on track_lin_vel_xy /
    track_ang_vel_z, names every recipe keeps)."""
    cfg.episode_length_s = episode_s
    cfg.commands.base_velocity.rel_standing_envs = standing
    cfg.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": tilt})
    cfg.curriculum.track_std = None
    if clock:
        cfg.observations.policy.gait_clock = ObsTerm(func=mdp.gait_clock, params={"period": clock})
        cfg.observations.critic.gait_clock = ObsTerm(func=mdp.gait_clock, params={"period": clock})
    if history:
        cfg.observations.policy.history_length = history


def _push_fixed(cfg, v, every_s):
    """A fixed push instead of the curriculum-driven one (the curriculum rewrites the range every level)."""
    cfg.curriculum.push_velocity = None
    cfg.events.push_robot.interval_range_s = (every_s, every_s)
    cfg.events.push_robot.params["velocity_range"] = {"x": (-v, v), "y": (-v, v)}


# ── #83 "Architecture Is All You Need" (G1, real): standing = double support in the phase-contact term, no standing ledger ──
@configclass
class Rewards83Cfg:
    track_lin_vel_xy, track_ang_vel_z = _track(1.0, 1.0, STD_WIDE)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"threshold": 1.0, "sensor_cfg": _NON_FEET})
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-0.2, params={"asset_cfg": _FEET, "sensor_cfg": _FEET_SENSOR})
    joint_deviation_hips = RewTerm(func=mdp.joint_deviation_l1, weight=-1.0, params={"asset_cfg": _HIPS})
    joint_deviation_torso = RewTerm(func=mdp.joint_deviation_l1, weight=-1.0, params={"asset_cfg": _TORSO})
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-0.5, params={"asset_cfg": _ARMS})
    feet_flat = RewTerm(func=mdp.feet_flat_bonus, weight=1.0, params={"asset_cfg": _FEET, "sensor_cfg": _FEET_SENSOR, "sigma_deg": 2.3})   # their 1 - exp(-25 x) as income
    base_height = RewTerm(func=mdp.base_height_l2, weight=-50.0, params={"target_height": H1_2_HEIGHT})
    phase_contact = RewTerm(func=mdp.feet_gait_stance_intent, weight=0.5,
                            params={"period": 0.8, "offset": [0.0, 0.5], "sensor_cfg": _FEET_SENSOR, "threshold": 0.55, "command_name": "base_velocity", "cmd_thr": 0.005})
    stand_still = RewTerm(func=mdp.joint_deviation_l1_standing, weight=-0.1, params={"asset_cfg": _BODY, "command_name": "base_velocity", "cmd_thr": 0.005})
    feet_clearance = RewTerm(func=mdp.foot_clearance_decay_cmd, weight=1.0,
                             params={"asset_cfg": _FEET, "target_height": 0.10, "foot_thickness": 0.02, "kappa": 5.0, "std": 0.05, "tanh_mult": 2.0, "command_name": "base_velocity"})
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)


def recipe_83(cfg):
    cfg.rewards = Rewards83Cfg()
    _common(cfg, episode_s=20.0, standing=0.05, clock=0.8, history=5)


# ── our ledger inverted: walking-first, standing as a corner case ──
def recipe_inverse(cfg):
    R = cfg.rewards
    for n in ("upright_bonus", "standing_pose", "torso_lin_vel_xy", "torso_ang_vel", "base_pos_xy_l2_from_spawn",
              "torso_stability_lin", "torso_stability_ang", "stance_bonus_legs_torso", "foot_stance_tracking", "capture_point_touchdown"):
        setattr(R, n, None)                                   # the nine standing terms + the balance-line step term
    R.alive.weight = 3.0
    R.track_lin_vel_xy.params["std"] = STD_WIDE; R.track_ang_vel_z.params["std"] = STD_WIDE
    R.track_ang_vel_z.weight = 20.0
    R.track_vel_hard = RewTerm(func=mdp.track_vel_hard, weight=5.0, params={"asset_cfg": _ROBOT, "command_name": "base_velocity"})
    R.low_speed = RewTerm(func=mdp.low_speed_penalty, weight=2.0, params={"asset_cfg": _ROBOT, "command_name": "base_velocity", "cmd_thr": 0.03})
    R.gait.func = mdp.feet_gait_stance_intent
    R.gait.params = {"period": 0.75, "offset": [0.0, 0.5], "sensor_cfg": _FEET_SENSOR, "threshold": 0.55, "command_name": "base_velocity", "cmd_thr": 0.03}
    R.feet_clearance = RewTerm(func=mdp.foot_clearance_decay_cmd, weight=6.0,
                               params={"asset_cfg": _FEET, "target_height": 0.10, "foot_thickness": 0.02, "kappa": 5.0, "std": 0.05, "tanh_mult": 2.0, "command_name": "base_velocity"})
    R.joint_deviation_hips.func = mdp.joint_deviation_l1
    R.joint_deviation_hips.params = {"asset_cfg": R.joint_deviation_hips.params["asset_cfg"]}
    R.feet_yaw_bonus = RewTerm(func=mdp.feet_yaw_bonus_turn_faded, weight=3.0, params={"asset_cfg": _FEET, "command_name": "base_velocity", "sigma_deg": 3.0, "wz_ref": 0.5})
    R.feet_flat_bonus = RewTerm(func=mdp.feet_flat_bonus, weight=1.0, params={"asset_cfg": _FEET, "sensor_cfg": _FEET_SENSOR, "sigma_deg": 5.0})
    R.feet_air_time = RewTerm(func=mdp.feet_air_time_touchdown, weight=0.5, params={"sensor_cfg": _FEET_SENSOR, "command_name": "base_velocity", "threshold": 0.25, "cmd_thr": 0.03})
    R.stand_still = RewTerm(func=mdp.joint_deviation_l1_standing, weight=-0.5, params={"asset_cfg": _BODY, "command_name": "base_velocity", "cmd_thr": 0.03})
    _common(cfg, episode_s=20.0, standing=0.05, clock=0.75, history=5)


# ── unitree_rl_gym H1-2 (the robot's own published walk; legged_gym H1_2RoughCfg + h1_2_env.py rewards) ──
@configclass
class RewardsUnitreeCfg:
    track_lin_vel_xy, track_ang_vel_z = _track(1.0, 0.5, STD_WIDE)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": H1_2_HEIGHT})
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    hip_pos = RewTerm(func=mdp.joint_deviation_l2, weight=-1.0, params={"asset_cfg": _HIPS})
    contact_no_vel = RewTerm(func=mdp.feet_slide, weight=-0.2, params={"asset_cfg": _FEET, "sensor_cfg": _FEET_SENSOR})
    feet_swing_height = RewTerm(func=mdp.feet_swing_height_penalty, weight=-20.0, params={"asset_cfg": _FEET, "sensor_cfg": _FEET_SENSOR, "target": 0.08})
    contact = RewTerm(func=mdp.feet_gait, weight=0.18,
                      params={"period": 0.8, "offset": [0.0, 0.5], "sensor_cfg": _FEET_SENSOR, "threshold": 0.55, "command_name": None})   # always on: steps in place at zero command
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-1.0, params={"asset_cfg": _ARMS})   # ours: their deploy holds the arms outside the policy


def recipe_unitree(cfg):
    cfg.rewards = RewardsUnitreeCfg()
    _common(cfg, episode_s=20.0, standing=0.0, clock=0.8, history=0)
    ev = cfg.events
    ev.physics_material.params["static_friction_range"] = (0.1, 1.25)
    ev.physics_material.params["dynamic_friction_range"] = (0.1, 1.25)
    ev.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
    _push_fixed(cfg, 1.5, 5.0)


# ── Menlo / Asimov: no clock, nine terms, per-joint tolerances, privileged critic, targeted DR ──
@configclass
class RewardsMenloCfg:
    track_lin_vel_xy, track_ang_vel_z = _track(2.0, 2.0, STD_WIDE)
    upright = RewTerm(func=mdp.upright_bonus_exp, weight=1.0, params={"asset_cfg": _ROBOT, "sigma_sq": 0.2})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    body_ang_vel = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.08)
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.03, params={"asset_cfg": _ROBOT})
    air_time = RewTerm(func=mdp.feet_air_time_touchdown, weight=0.5, params={"sensor_cfg": _FEET_SENSOR, "command_name": "base_velocity", "threshold": 0.25, "cmd_thr": 0.03})
    contact_force = RewTerm(func=mdp.contact_forces, weight=-0.01, params={"threshold": 1000.0, "sensor_cfg": _FEET_SENSOR})
    pose = RewTerm(func=mdp.stance_bonus_vec, weight=1.0, params={"asset_cfg": _LEGS_TORSO, "std_map": dict(MENLO_STD), "std_default": 0.3})
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-0.5, params={"asset_cfg": _ARMS})   # ours: Asimov has no arms in the policy


def recipe_menlo(cfg):
    cfg.rewards = RewardsMenloCfg()
    _common(cfg, episode_s=20.0, standing=0.05, clock=None, history=0)
    gene_dr_all(cfg)
    gene_pc_contacts(cfg)


# ── humanoid-gym (XBot-L): clock 0.64 s, 15-frame history, a reference swing pose, a hip wall ──
@configclass
class RewardsHGymCfg:
    joint_pos = RewTerm(func=mdp.joint_pos_ref_swing, weight=1.6, params={"asset_cfg": _SWING_LEGS, "period": 0.64, "scale": 0.17})
    feet_contact_number = RewTerm(func=mdp.feet_contact_number, weight=1.2, params={"sensor_cfg": _FEET_SENSOR, "period": 0.64, "threshold": 0.1})
    track_lin_vel_xy, track_ang_vel_z = _track(1.2, 1.1, STD_HGYM)
    feet_clearance = RewTerm(func=mdp.foot_clearance_reward_cmd, weight=1.0,
                             params={"asset_cfg": _FEET, "target_height": 0.06, "std": 0.05, "tanh_mult": 2.0, "command_name": "base_velocity", "cmd_thr": 0.1})   # their binary |h - .06| < .01 as a kernel
    feet_air_time = RewTerm(func=mdp.feet_air_time_touchdown, weight=1.0, params={"sensor_cfg": _FEET_SENSOR, "command_name": "base_velocity", "threshold": 0.0, "cmd_thr": 0.03})
    orientation = RewTerm(func=mdp.upright_bonus_exp, weight=1.0, params={"asset_cfg": _ROBOT, "sigma_sq": 0.05})
    default_joint_pos = RewTerm(func=mdp.hip_wall_bonus, weight=0.5, params={"asset_cfg": _HIPS, "all_cfg": _BODY, "deadband": 0.1})
    vel_mismatch = RewTerm(func=mdp.vel_mismatch_exp, weight=0.5, params={"asset_cfg": _ROBOT})
    track_vel_hard = RewTerm(func=mdp.track_vel_hard, weight=0.5, params={"asset_cfg": _ROBOT, "command_name": "base_velocity"})
    low_speed = RewTerm(func=mdp.low_speed_penalty, weight=0.2, params={"asset_cfg": _ROBOT, "command_name": "base_velocity", "cmd_thr": 0.1})
    base_height = RewTerm(func=mdp.base_height_exp, weight=0.2, params={"asset_cfg": _ROBOT, "target": H1_2_HEIGHT, "gain": 100.0})
    foot_slip = RewTerm(func=mdp.feet_slide, weight=-0.05, params={"asset_cfg": _FEET, "sensor_cfg": _FEET_SENSOR})
    feet_contact_forces = RewTerm(func=mdp.feet_contact_forces_hinge, weight=-0.01, params={"sensor_cfg": _FEET_SENSOR, "max_force": 700.0, "cap": 400.0})
    action_smoothness = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-4)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    collision = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"threshold": 1.0, "sensor_cfg": _NON_FEET})
    joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-0.5, params={"asset_cfg": _ARMS})   # ours


def recipe_hgym(cfg):
    cfg.rewards = RewardsHGymCfg()
    _common(cfg, episode_s=24.0, standing=0.0, clock=0.64, history=15)
    ev = cfg.events
    ev.physics_material.params["static_friction_range"] = (0.1, 2.0)
    ev.physics_material.params["dynamic_friction_range"] = (0.1, 2.0)
    ev.add_base_mass.params["mass_distribution_params"] = (-5.0, 5.0)
    _push_fixed(cfg, 0.2, 4.0)


# ── the five tasks ──
def _lm8(cfg, recipe):
    _lm7u_stack(cfg)
    gene_lm7d_base(cfg)
    recipe(cfg)
    _apply_overrides(cfg, _load_overrides())


@configclass
class RobotEnvCfgLM8_83(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _lm8(self, recipe_83)


@configclass
class RobotPlayEnvCfgLM8_83(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _lm8(self, recipe_83)


@configclass
class RobotEnvCfgLM8_Inverse(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _lm8(self, recipe_inverse)


@configclass
class RobotPlayEnvCfgLM8_Inverse(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _lm8(self, recipe_inverse)


@configclass
class RobotEnvCfgLM8_Unitree(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _lm8(self, recipe_unitree)


@configclass
class RobotPlayEnvCfgLM8_Unitree(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _lm8(self, recipe_unitree)


@configclass
class RobotEnvCfgLM8_Menlo(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _lm8(self, recipe_menlo)


@configclass
class RobotPlayEnvCfgLM8_Menlo(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _lm8(self, recipe_menlo)


@configclass
class RobotEnvCfgLM8_HGym(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _lm8(self, recipe_hgym)


@configclass
class RobotPlayEnvCfgLM8_HGym(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _lm8(self, recipe_hgym)
