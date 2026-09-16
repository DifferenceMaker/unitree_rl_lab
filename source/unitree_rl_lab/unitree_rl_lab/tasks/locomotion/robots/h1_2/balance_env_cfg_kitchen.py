"""Kitchen — the p14_kitchen recipe promoted to parent + trunk (operator 2026-09-16:
"Make kitchen the absolute parent and trunk its reward terms. MAKE SURE that they
are totally the delta runs that the kitchen was trained on. Then name the next
subphase runs as p14b_*").

Balance-QIK-FIXLR + the p14_kitchen.job apply_edits baked 1:1, in the job's order,
values from the harvested p14_kitchen_2026-09-13 milestone (params/env.yaml, 39
reward rows, 37 non-zero). Job overrides are applied LAST, so a p14b job is a pure
delta on top of exactly what kitchen trained on. Same 87-obs contract and the same
runner cfg (QueueFixedLRPPORunnerCfg: fixed lr 3e-4, GuardedPPO), so p14_kitchen
warmstarts stay valid and harvest paths are unchanged.

Hardware 2026-09-14/16: kitchen is the only policy clearing both lean complaints,
stands at 0.0..0.7 deg pitch with zero drift, but (a) leans forward in proportion
to arm extension (+1.7 deg at the start pose, 0.3 deg short of the ~2 deg stepping
boundary) and (b) walks the LEFT foot inward once the policy engages (FixStand is
symmetric). p14b attacks both: upright dose, stance hinge, the 790 g hand plant,
exploration reset. Nothing in this file is a delta — deltas live in the jobs.

The trunk BODY is the real-hand plant (operator 2026-09-16: "Make sure the model is
the correct one with the correct hand"): h1_2_comx06_hand790.urdf -- see _use_real_hand.
Every p14b run trains on it; p14b_plant (zero reward delta) is the same-plant reference
and p14_kitchen (comx06, 0.19 kg hands) the cross-plant baseline.

Trunk parity is proven by scripts/queue/lineage_check_p14b.py (aspired-isaac-lab):
smoke env.yaml vs the kitchen milestone env.yaml, all sections (asset basename expected
to differ: hand790).
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .balance_env_cfg_queue_ik import _swap_in_ik_command


def _make_kitchen(cfg):
    """The p14_kitchen.job apply_edits, baked. Order and values as in the job."""
    R = cfg.rewards

    # ── p13g_sqfixshape recipe (the p14 parent) ──
    R.foot_stance_tracking.params["nominal_foot_pos_b"] = [[0.0, 0.13], [0.0, -0.13]]
    R.feet_too_near.params["threshold"] = 0.22
    cfg.events.micro_push = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(1.0, 3.0),
        params={"velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
    )
    cfg.commands.arm_pose_command.orientation_mode = True
    R.feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward, weight=5.0,
        params={"std": 0.05, "tanh_mult": 2.0, "target_height": 0.15,
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])},
    )
    R.feet_air_time_step.weight = -1.0
    R.gait = RewTerm(
        func=mdp.feet_gait_recovery, weight=0.5,
        params={"period": 0.75, "offset": [0.0, 0.5], "threshold": 0.55, "gate_speed": 0.15,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )
    cfg.scene.contact_forces.history_length = 4
    R.base_height.weight = 0.0
    R.knee_torque_cap = RewTerm(
        func=mdp.joint_torque_over_limit, weight=-0.2,
        params={"limit_nm": 100.0, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])},
    )
    cfg.actions.JointPositionAction.clip = None
    R.action_cap = RewTerm(
        func=mdp.action_magnitude_over, weight=-0.05,
        params={"threshold": 1.5, "square": True, "max_excess": 5.0},
    )
    R.feet_too_far = RewTerm(
        func=mdp.feet_too_far, weight=-8.0,
        params={"threshold": 0.45, "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])},
    )
    R.feet_crossed = RewTerm(func=mdp.feet_crossed, weight=-8.0, params={"min_gap": 0.05})

    # ── SPLIT torso_stability (p13h_split): the combined term is a PRODUCT of the linear
    # and angular kernels and paid ~17%; split, each axis can be climbed on its own. ──
    R.torso_stability_bonus.weight = 0.0
    R.torso_stability_lin = RewTerm(
        func=mdp.torso_stability_bonus, weight=3.0,
        params={"std_lin": 0.15, "std_ang": 1000000.0,
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    R.torso_stability_ang = RewTerm(
        func=mdp.torso_stability_bonus, weight=3.0,
        params={"std_lin": 1000000.0, "std_ang": 0.30,
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )

    # ── upright CLIFF widened (std 0.01 -> 0.025), weight LEFT at 1.0 (w 3.0 competed with
    # stillness in p13h_stillup). p14b_tilt is the run that raises both. ──
    R.upright_bonus.params["std"] = 0.025

    # ── the COUNTERBALANCE stimulus: pay for hand position on every non-default arm draw ──
    R.desk_reach = RewTerm(func=mdp.desk_reach_bonus, weight=4.0,
                           params={"sigma": 0.25, "require_desk_draw": False})

    # ── energy at a dose that can be seen (~1% of return) ──
    R.energy = RewTerm(func=mdp.energy, weight=-0.01, params={})
    R.joint_vel_reg = RewTerm(func=mdp.joint_vel_l2, weight=-0.01, params={})


def _use_real_hand(cfg):
    """The trunk BODY: comx06 with the real 790 g Inspire hand. Training carried a 0.192 kg
    vendor-lineage hand + a 0.324 kg wrist stand-in; the real RH56DFTP is 790 g on a
    0.124 kg wrist. h1_2_comx06_hand790.urdf (aspired-isaac-lab assets/robot/h1_2,
    MODELS.md) keeps total mass 77.2676 kg and the whole-body CoM. Hardware 2026-09-16:
    kitchen leans forward in proportion to arm extension -- the heavier-than-modelled
    hand is the prime suspect. sim2sim body: scene_comx06_armature_hand790.xml
    (run_mujoco_sim.sh --hand790). Fails loudly if the base body is not comx06, so a
    wrong ROBOT_ASSETS_DIR can never train silently on the old hand."""
    spawn = cfg.scene.robot.spawn
    spawn.asset_path = spawn.asset_path.replace("h1_2_comx06.urdf", "h1_2_comx06_hand790.urdf")
    assert spawn.asset_path.endswith("h1_2_comx06_hand790.urdf"), (
        f"[Kitchen trunk] expected the comx06 body to swap to hand790, got {spawn.asset_path}")


@configclass
class RobotEnvCfgKitchen(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _swap_in_ik_command(self)
        _make_kitchen(self)
        _use_real_hand(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgKitchen(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _swap_in_ik_command(self)
        _make_kitchen(self)
        _use_real_hand(self)
        self.commands.arm_pose_command.debug_vis = True
        # Play at the full workspace immediately (as Balance-QIK's play cfg does).
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
