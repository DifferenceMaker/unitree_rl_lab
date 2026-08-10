"""Balance-Desk5 — the dp5 TRUNK as a task, not as job restatement.

Every conclusion dp4/dp4b/dp4c settled is baked in here, so a dp5 job carries
ONLY its own axis. Restating a 25-line contract in every job is how the p13
arm_pose accident happened; this removes the restatement surface entirely.

A NEW task id (not a change to Balance-Desk) on purpose: the historical desk
policies (dp3/dp4/dp4b/dp4c) replay their previews against Balance-Desk, and
mutating that task under them would silently change what those videos show
(a desk collider would appear, the ladder would move). Additive, like
Balance-QIK was.

WHAT THE TRUNK CONTAINS (and the dp4 evidence for each):
  * gait clock, anchor-gated       — gaitanchor was the operator favourite and
                                     the phasing visibly transferred to hardware
  * cheap stepping (feet_air -1,   — the sustained-push gene AND the
    foot_displacement 0)             anti-limit-riding lever (dof_pos_limits
                                     -0.089 -> -0.035 when it is added)
  * anchor_hold sigma 0.15 +       — 0.08 RETIRED: it bought pelvis-reach
    heading_to="spawn"               contortion (squat / "lean in weird ways");
                                     spawn-yaw = table parallelism, not bearing
  * anchor wander v3: half-normal  — dp4c's linear [0,15] put 44.5% of jumps
    sigma 3 cm, clamp 10 cm          over 5 cm and 21.8% over 8 cm, every 1-2 s
  * desk slab COUPLED to the       — the anchor IS the desk centre (4-marker
    anchor + desk_hit on wrists       midpoint); reset-only placement decoupled
                                     them. deskcol proved desk physics alone
                                     already buys collision robustness
  * desk arm draws 20/30/50 held   — dp4c resampled every 2-10 s while the arm
    8-15 s                           needs ~2.5 s just to travel at 0.6 rad/s:
                                     far targets vanished before arrival
  * ladder 1.0 m/s / 30 N, sparse  — operator's reduced ceiling; still big
    (20-30 s / 30-50 s)              enough to shove the robot INTO the desk,
                                     which is a real situation to train
  * heading income widened         — std 0.1 was dead past ~0.2 rad (exp(-9))
    (heading_stable_bonus std 0.25)   while the logs showed a CHRONIC 0.6-0.9 rad
                                     yaw debt
  * hip-roll 80 Nm guard           — inherited burnout insurance

DELIBERATELY OUT: jacc of any size (parked), sigma 0.08, stood_ground,
unbounded heading L2.
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_desk import _make_desk
from .balance_env_cfg_queue import _apply_overrides, _load_overrides

ANCHOR_FWD = 0.5      # anchor = spawn + this, along the spawn heading
DESK_TOP_Z = 1.0      # the real desk height
DESK_THICK = 0.05


def _make_desk5(cfg):
    """Apply the dp5 trunk on top of the desk-line world."""
    # --- parent genes (dp4c_triple lineage) ---
    cfg.rewards.foot_stance_tracking.params["nominal_foot_pos_b"] = ([0.0, 0.13], [0.0, -0.13])
    cfg.rewards.foot_stance_tracking.weight = 3.0
    cfg.rewards.feet_too_near.params["threshold"] = 0.22
    cfg.rewards.base_forward_zone.weight = -6.0
    cfg.rewards.hiproll_over_cap = RewTerm(
        func=mdp.joint_torque_over_limit, weight=-0.2,
        params={"limit_nm": 80.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )
    cfg.commands.arm_pose_command.workspace_offset = ((-0.05, -0.15, -0.15), (0.55, 0.35, 0.85))

    # --- cheap stepping: the sustained-push gene + the anti-limit-riding lever ---
    cfg.rewards.feet_air_time_step.weight = -1.0
    cfg.rewards.foot_displacement_l2_from_spawn.weight = 0.0

    # --- anchor: obs, hold kernel (sigma 0.15, spawn-yaw heading), gait clock ---
    cfg.observations.policy.anchor_point = ObsTerm(
        func=mdp.anchor_point_b,
        params={"fwd_offset": ANCHOR_FWD, "height_w": DESK_TOP_Z, "noise_std": 0.02},
    )
    cfg.observations.critic.anchor_point = ObsTerm(
        func=mdp.anchor_point_b,
        params={"fwd_offset": ANCHOR_FWD, "height_w": DESK_TOP_Z, "noise_std": 0.0},
    )
    cfg.rewards.anchor_hold = RewTerm(
        func=mdp.anchor_hold_bonus, weight=8.0,
        params={"fwd_offset": ANCHOR_FWD, "heading_scale": 0.5,
                "sigma": 0.15, "heading_to": "spawn"},
    )
    cfg.rewards.gait = RewTerm(
        func=mdp.feet_gait_recovery, weight=0.5,
        params={"period": 0.75, "offset": [0.0, 0.5], "threshold": 0.55,
                "gate_speed": 0.15, "gate_anchor_dist": 0.15,
                "sensor_cfg": SceneEntityCfg("contact_forces",
                                             body_names=[".*_ankle_roll_link"])},
    )

    # --- anchor motion v3 (half-normal 3 cm, clamp 10) + slow yaw re-point ---
    cfg.events.move_anchor = EventTerm(
        func=mdp.move_anchor, mode="interval", interval_range_s=(1.0, 2.0),
        params={"radius_range": (0.0, 0.10), "yaw_range": None,
                "half_normal_sigma": 0.03},
    )
    cfg.events.anchor_repoint = EventTerm(
        func=mdp.move_anchor, mode="interval", interval_range_s=(4.0, 8.0),
        params={"radius_range": (0.0, 0.0), "yaw_range": (-0.4, 0.4)},
    )

    # --- heading income over a useful range ---
    cfg.rewards.heading_stable_bonus.params["std"] = 0.25

    # --- THE DESK, coupled to the anchor it is defined by ---
    cfg.scene.desk = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 2.0, DESK_THICK),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.25, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(ANCHOR_FWD, 0.0, DESK_TOP_Z - DESK_THICK * 0.5)),
    )
    # filter rule: ONE body per env on the sensor, and EVERY filter entry must
    # resolve to exactly one prim per env — a regex matching several is rejected
    # silently and the penalty pays 0 forever (cost us three inert terms).
    cfg.scene.desk_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
                                "{ENV_REGEX_NS}/Robot/right_wrist_yaw_link"],
        update_period=0.0,
    )
    _desk_params = {"fwd_offset": ANCHOR_FWD, "top_z": DESK_TOP_Z, "thickness": DESK_THICK}
    cfg.events.place_desk = EventTerm(func=mdp.place_desk, mode="reset", params=_desk_params)
    cfg.events.place_desk_follow = EventTerm(
        func=mdp.place_desk, mode="interval", interval_range_s=(1.0, 2.0), params=_desk_params,
    )
    cfg.rewards.desk_hit = RewTerm(
        func=mdp.desk_hit_penalty, weight=-1.0,
        params={"force_thr": 1.0, "max_val": 10.0},
    )

    # --- desk-level arm work, held long enough to matter ---
    c = cfg.commands.arm_pose_command
    c.default_pose_prob = 0.20
    c.desk_level_prob = 0.30
    c.desk_fwd_offset = ANCHOR_FWD
    c.desk_height_w = DESK_TOP_Z
    c.desk_x_range = (-0.20, 0.20)
    c.desk_y_range = (-0.50, 0.50)
    c.desk_z_jitter = 0.05
    c.resampling_time_range = (8.0, 15.0)

    # --- reduced push ladder, sparse cadence ---
    cfg.curriculum.push_velocity.params["levels"] = (0.25, 0.5, 0.75, 1.0)
    cfg.curriculum.sustained_push.params["levels"] = (
        ((0.0, 0.0), (0.0, 0.0)),
        ((0.0, 10.0), (1.5, 3.0)),
        ((0.0, 20.0), (2.0, 3.5)),
        ((0.0, 30.0), (2.0, 4.0)),
    )
    cfg.events.push_robot.interval_range_s = (20.0, 30.0)
    cfg.events.sustained_push_apply.interval_range_s = (30.0, 50.0)


@configclass
class RobotEnvCfgDesk5(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_desk(self)
        _make_desk5(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgDesk5(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_desk(self)
        _make_desk5(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
