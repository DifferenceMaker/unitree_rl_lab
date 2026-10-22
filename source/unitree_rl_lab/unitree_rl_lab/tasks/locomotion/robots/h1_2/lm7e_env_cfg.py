"""lm7e (2026-10-22): the walking-recipe survey as single-axis genes on the lm7d keepers.

Parents (operator sim2sim 2026-10-21): A = lm7d_turn_dead03 @91399 (the only honest turn: planted foot + stepping foot), B =
lm7d_torsowalk_near26 @90975 ("Best small steps ive seen in this batch. coordinated and symmetrical"). Every row re-applies its
parent's genes on the LM7U task (gene_lm7d_base + gene_turn_dead03 / gene_torsowalk_near26) and adds ONE lever, except the kits.

Group A (notes/walking_recipes_vs_lm7_2026-10-21.md, section 5): hips (always-on hip price, 4 recipes), footbonus (foot orientation as
INCOME -- operator: a reward, not a penalty), clear08 (clearance target .15 -> .08: ours was the highest in the set), yaw20 (yaw
tracking >= linear + yaw pushes), dofcap8 (the stop as a wall), airtime (Menlo), gforce (Menlo), angmom (#333 / Menlo + the torso
blueprint), posture (Menlo's per-joint tolerances), stand5 (#83: standing = double support inside the gait term), alive3 (the 15 %
rule), ladder10 (wide kernel + a low-speed term), resets (G1 example), kit = hips + footbonus + clear08.
Group B (targeted DR, Menlo's philosophy -- randomise what varies, not what was measured; operator 2026-10-22: no encoder band,
PD on, soles off, friction narrow, obs and action delay 0-1 step, resets widened, joint-velocity and IMU noise trimmed, CoM as an
axis, mass off). Group C (privileged critic: the critic sees contact, the actor does not; deploy contract untouched).
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg_hipsarm_dr import install_dr_hooks
from .lm7u_env_cfg import _FEET, gene_lm7d_base, gene_torsowalk_near26, gene_turn_dead03

_FEET_SENSOR = SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])
PARENT_A, IT_A = "lm7d_turn_dead03_2026-10-20", 91399
PARENT_B, IT_B = "lm7d_torsowalk_near26_2026-10-20", 90975
MENLO_STD = {"hip_pitch": 0.5, "hip_roll": 0.25, "hip_yaw": 0.2, "knee": 0.5, "ankle_pitch": 0.2, "ankle_roll": 0.12, "torso": 0.2}


# ── parents ──
def parent_A(cfg):
    gene_lm7d_base(cfg); gene_turn_dead03(cfg)


def parent_B(cfg):
    gene_lm7d_base(cfg); gene_torsowalk_near26(cfg)


# ── Group A ──
def gene_hips(cfg):
    """joint_deviation_hips: the turn-faded L1 -> the plain always-on L1 at the same -2 (Unitree H1-2, G1 example, #83, humanoid-gym
    all price hip roll/yaw at every command; the fade was an lm5e precaution, affordable to drop: a 0.3 rad excursion on both hips
    costs 2 x 0.6 = 1.2/step against a yaw income of 14-20)."""
    t = cfg.rewards.joint_deviation_hips
    assert t.func is mdp.joint_deviation_l1_turn_gated and t.weight == -2.0, "[hips] from the trunk's turn-faded -2"
    t.func = mdp.joint_deviation_l1
    t.params = {"asset_cfg": t.params["asset_cfg"]}


def gene_footbonus(cfg):
    """Foot orientation as income: feet_yaw_bonus +3 (L1 inside the exp, sigma 3 deg, turn-faded) + feet_flat_bonus +1 (roll + pitch of
    the feet in contact, sigma 5 deg). The lm7d footyaw L2 was flat at the bottom; HOMIE / #83 price orientation with gradient."""
    R = cfg.rewards
    assert getattr(R, "feet_yaw", None) is None and getattr(R, "feet_yaw_bonus", None) is None, "[footbonus] no foot-orientation term underneath"
    R.feet_yaw_bonus = RewTerm(func=mdp.feet_yaw_bonus_turn_faded, weight=3.0,
                               params={"asset_cfg": _FEET, "command_name": "base_velocity", "sigma_deg": 3.0, "wz_ref": 0.5})
    R.feet_flat_bonus = RewTerm(func=mdp.feet_flat_bonus, weight=1.0,
                                params={"asset_cfg": _FEET, "sensor_cfg": _FEET_SENSOR, "sigma_deg": 5.0})


def gene_clear08(cfg):
    """Clearance target .15 -> .08 on the plain term (and on the swing bonus when the parent has one): Unitree .08, humanoid-gym .06, G1 .10."""
    R = cfg.rewards
    assert R.feet_clearance.params["target_height"] == 0.15 and R.feet_clearance.weight == 6.0, "[clear08] from clear6 @ .15"
    R.feet_clearance.params["target_height"] = 0.08
    if getattr(R, "feet_swing", None) is not None:
        R.feet_swing.params["target_height"] = 0.08


def gene_yaw20(cfg):
    """track_ang_vel_z 14 -> 20 (= linear; #333 3 vs 2, HOMIE 2 vs 1.5, X-Loco 5 = 5) + a yaw push event +-0.5 rad/s every 5-8 s
    (X-Loco +-0.78). Its own event: push_velocity_curriculum rewrites push_robot's x/y range every level and would erase a yaw key."""
    assert cfg.rewards.track_ang_vel_z.weight == 14.0, "[yaw20] from 14"
    cfg.rewards.track_ang_vel_z.weight = 20.0
    cfg.events.push_yaw = EventTerm(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(5.0, 8.0),
                                    params={"velocity_range": {"yaw": (-0.5, 0.5)}})


def gene_dofcap8(cfg):
    assert cfg.rewards.dof_pos_limits.weight == -100.0, "[dofcap8] from the trunk's -100"
    cfg.rewards.dof_pos_limits.weight = -800.0


def gene_airtime(cfg):
    """Menlo air_time +0.5: (t_air - 0.25) at touchdown under a command (0.5 would punish our 0.34 s swing)."""
    assert getattr(cfg.rewards, "feet_air_time", None) is None
    cfg.rewards.feet_air_time = RewTerm(func=mdp.feet_air_time_touchdown, weight=0.5,
                                        params={"sensor_cfg": _FEET_SENSOR, "command_name": "base_velocity", "threshold": 0.25, "cmd_thr": 0.03})


def gene_gforce(cfg):
    """Menlo contact-force limit: the Isaac contact_forces hinge above 1000 N (~1.5 body weights) per foot, -0.01."""
    cfg.rewards.feet_force_cap = RewTerm(func=mdp.contact_forces, weight=-0.01, params={"threshold": 1000.0, "sensor_cfg": _FEET_SENSOR})


def gene_angmom(cfg):
    """The torso-pitch blueprint: angular momentum ||L||^2 -0.3 (1.5 % of tracking, #333 / Menlo), base_height -15 -> -30 (#83 -50),
    arm swing 1 -> 2 at amplitude .25 -> .40 (anti-phase arms cancel what the legs otherwise absorb as torso pitch)."""
    R = cfg.rewards
    R.angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.3, params={"asset_cfg": SceneEntityCfg("robot")})
    assert R.base_height.weight == -15.0, "[angmom] base_height from -15"
    R.base_height.weight = -30.0
    assert R.arm_swing.weight == 1.0 and R.arm_swing.params["amplitude"] == 0.25, "[angmom] arm_swing from 1 @ .25"
    R.arm_swing.weight = 2.0
    R.arm_swing.params["amplitude"] = 0.4


def gene_posture(cfg):
    """stance_bonus (one std .15 over 13 joints = dead past 0.3 rad) -> stance_bonus_vec with Menlo's per-joint tolerances; same weight."""
    t = cfg.rewards.stance_bonus_legs_torso
    assert t.func is mdp.stance_bonus and t.params["std"] == 0.15 and t.weight == 0.75, "[posture] from stance_bonus std .15 @ .75"
    t.func = mdp.stance_bonus_vec
    t.params = {"asset_cfg": t.params["asset_cfg"], "std_map": dict(MENLO_STD), "std_default": 0.3}


def gene_stand5(cfg):
    """#83: standing share 25 % -> 5 % and the gait term pays double support at a zero command (stance intent), so standing is a
    corner case of walking, not a task with its own ledger."""
    C = cfg.commands.base_velocity
    assert C.rel_standing_envs == 0.25, "[stand5] from 25 % standing envs"
    C.rel_standing_envs = 0.05
    g = cfg.rewards.gait
    assert g.func is mdp.feet_gait and g.weight == 3.0, "[stand5] from feet_gait 3"
    g.func = mdp.feet_gait_stance_intent
    g.params = {"period": g.params["period"], "offset": g.params["offset"], "sensor_cfg": g.params["sensor_cfg"],
                "threshold": g.params["threshold"], "command_name": "base_velocity", "cmd_thr": g.params.get("cmd_thr", 0.03)}


def gene_alive3(cfg):
    """alive 30 -> 3: 15 % of the linear tracking ceiling (legged_gym, G1 example, #83) instead of 150 %."""
    assert cfg.rewards.alive.weight == 30.0, "[alive3] from 30"
    cfg.rewards.alive.weight = 3.0


def gene_ladder10(cfg):
    """The tracking ladder stops at .10 (never .05) and humanoid-gym's low_speed term (+0.2) supplies the slow-command fidelity."""
    cs = cfg.curriculum.track_std
    assert cs.params["levels"] == (0.15, 0.10, 0.05), "[ladder10] from the .15/.10/.05 ladder"
    cs.params["levels"] = (0.15, 0.10)
    cfg.rewards.low_speed = RewTerm(func=mdp.low_speed_penalty, weight=0.2,
                                    params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "cmd_thr": 0.03})


def gene_resets(cfg):
    """G1 example resets: spawn +-0.5 m / full yaw, joint pose +-0.05 rad and velocity +-1 rad/s at reset (reset_joints_by_offset --
    the trunk's reset_joints_by_scale scales a zero default velocity, so its velocity_range can never inject motion), and a tilt
    termination at 0.8 rad (46 deg)."""
    rb = cfg.events.reset_base.params["pose_range"]
    assert tuple(rb["x"]) == (-0.1, 0.1), "[resets] from +-0.1 m"
    rb["x"] = (-0.5, 0.5); rb["y"] = (-0.5, 0.5); rb["yaw"] = (-3.14, 3.14)
    cfg.events.reset_robot_joints = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                                              params={"position_range": (-0.05, 0.05), "velocity_range": (-1.0, 1.0)})
    cfg.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


def gene_kit(cfg): gene_hips(cfg); gene_footbonus(cfg); gene_clear08(cfg)


# ── Group B: targeted DR ──
def gene_dr_pd(cfg):
    ev = cfg.events.randomize_motor_strength
    assert tuple(ev.params["stiffness_distribution_params"]) == (1.0, 1.0), "[dr_pd] from nominal gains"
    ev.params["stiffness_distribution_params"] = (0.9, 1.1)
    ev.params["damping_distribution_params"] = (0.9, 1.1)


def gene_dr_fric(cfg):
    ev = cfg.events.physics_material
    assert tuple(ev.params["static_friction_range"]) == (1.0, 1.0), "[dr_fric] from fixed 1.0"
    ev.params["static_friction_range"] = (0.8, 1.2)
    ev.params["dynamic_friction_range"] = (0.8, 1.2)


def gene_dr_obsdelay(cfg):
    """Joint positions and velocities the ACTOR sees are delayed 0 or 1 control step, drawn per episode (Menlo 0-2 steps per CAN
    group at a faster loop; our measured obs age ~4 ms + the unmeasured per-joint skew). The critic keeps the true state."""
    P = cfg.observations.policy
    assert P.joint_pos_rel.func in (mdp.joint_pos_rel, mdp.joint_pos_rel_encoder) and P.joint_vel_rel.func is mdp.joint_vel_rel, "[dr_obsdelay] clean joint terms underneath"
    P.joint_pos_rel.func = mdp.joint_pos_rel_delayed
    P.joint_pos_rel.params["max_delay"] = 1
    P.joint_vel_rel.func = mdp.joint_vel_rel_delayed
    P.joint_vel_rel.params["max_delay"] = 1


def gene_dr_actdelay(cfg):
    """The applied target lags 0 or 1 policy step, drawn per episode (the p16_dr lag gene: RobustJointPositionAction, same affine)."""
    install_dr_hooks(cfg)
    cfg.actions.JointPositionAction.max_lag_steps = 1


def gene_dr_qdnoise(cfg):
    n = cfg.observations.policy.joint_vel_rel.noise
    assert n.n_min == -1.5 and n.n_max == 1.5, "[dr_qdnoise] from +-1.5 rad/s"
    n.n_min, n.n_max = -0.3, 0.3


def gene_dr_imunoise(cfg):
    n = cfg.observations.policy.base_ang_vel.noise
    assert n.n_min == -0.2 and n.n_max == 0.2, "[dr_imunoise] from +-0.2 rad/s"
    n.n_min, n.n_max = -0.05, 0.05


def gene_dr_com(cfg):
    assert getattr(cfg.events, "randomize_torso_com", None) is None
    cfg.events.randomize_torso_com = EventTerm(func=mdp.randomize_rigid_body_com, mode="startup",
                                               params={"com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025)},
                                                       "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"])})


def gene_dr_all(cfg):
    gene_dr_pd(cfg); gene_dr_fric(cfg); gene_dr_actdelay(cfg); gene_dr_obsdelay(cfg)   # actdelay installs the hooks on clean terms, obsdelay after
    gene_dr_qdnoise(cfg); gene_dr_imunoise(cfg); gene_dr_com(cfg)


# ── Group C: privileged critic (appended LAST in the critic group; the actor's 90 inputs untouched) ──
def gene_pc_contacts(cfg):
    Cr = cfg.observations.critic
    assert getattr(Cr, "feet_contact_flags", None) is None, "[pc] no feet terms in the critic underneath"
    Cr.feet_contact_flags = ObsTerm(func=mdp.feet_contact_flags, params={"sensor_cfg": _FEET_SENSOR})
    Cr.feet_force_z = ObsTerm(func=mdp.feet_force_z, params={"sensor_cfg": _FEET_SENSOR})
    Cr.feet_height = ObsTerm(func=mdp.feet_height, params={"asset_cfg": _FEET})
    Cr.feet_xy_speed = ObsTerm(func=mdp.feet_xy_speed, params={"asset_cfg": _FEET})
    Cr.feet_air_time_obs = ObsTerm(func=mdp.feet_air_time_obs, params={"sensor_cfg": _FEET_SENSOR})


def gene_pc_all(cfg): gene_dr_all(cfg); gene_pc_contacts(cfg)


# ── Group C, second wave (operator 2026-10-22: "we really need to push on the experimentation of giving the critic more information") ──
def gene_C_feet(cfg):
    """pc_contacts + the two hardware-wear terms (contact-force limit, air time) in one row: the critic sees the feet while the
    ledger prices stomping and shuffling. Warmstart = the pc10pad transplant."""
    gene_pc_contacts(cfg); gene_gforce(cfg); gene_airtime(cfg)


def gene_pc_plant(cfg):
    """dr_all + the privileged feet + the PLANT the DR drew, as critic inputs (7 more, appended after the feet terms): gain scale (kp, kd),
    friction, torso CoM offset (x, y), action lag, observation delay. The actor still sees only what the robot measures. Warmstart =
    the pc17pad transplant (critic 120 -> 137). Not in: the push (the critic already sees base_lin_vel, which carries it)."""
    gene_dr_all(cfg); gene_pc_contacts(cfg)
    Cr = cfg.observations.critic
    Cr.plant_gain_scale = ObsTerm(func=mdp.plant_gain_scale, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "torso_joint"])})
    Cr.plant_friction = ObsTerm(func=mdp.plant_friction, params={"asset_cfg": SceneEntityCfg("robot")})
    Cr.plant_com_offset = ObsTerm(func=mdp.plant_com_offset, params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]), "scale": 10.0})
    Cr.plant_action_lag = ObsTerm(func=mdp.plant_action_lag, params={"action_name": "JointPositionAction"})
    Cr.plant_obs_delay = ObsTerm(func=mdp.plant_obs_delay)
