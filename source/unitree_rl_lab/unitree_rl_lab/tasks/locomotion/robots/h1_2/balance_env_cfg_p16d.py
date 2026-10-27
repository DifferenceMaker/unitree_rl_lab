"""p16d (2026-10-27): the HANDOVER on the balance line -- activation DR genes on the stancewide recipe.

The homing rig measured the balance policy engaging from the walk policy's stance (unitree_mujoco/mujoco_sim/results/
HOMING_RESULTS_2026-10-27.md, sections 5-8): after the switch the robot shifts 7 cm RIGHT on average (max 15) and the heading kicks
+-3 deg in the first seconds, while the commander had placed it within 2.6 cm / 2 deg. The balance line has never seen a walk stance
at reset (trunk reset: default joints, +-0.1 m / +-0.5 rad spawn, zero velocity). Three levers, one per row, each the balance-line
mirror of an lm7f activation gene (lm7f_env_cfg.py), on the RECIPE that takes a warmstart (p16c: Balance-HipsArm-DR + feet_too_near
.28 + nominal +-.15 from p14g_hips_armfull @75600 -- the p14h_stancewide checkpoint itself detonated as a parent 4/4 in p16b):

  gene_resets      the lm7e/lm7f 'resets' keeper: spawn +-0.5 m / full yaw, joints +-0.05 rad / +-1 rad/s at reset
                   (reset_joints_by_offset -- the trunk's reset_joints_by_scale scales a zero default velocity and can never inject
                   motion), tilt termination 0.8 rad (46 deg)
  gene_activation  lm7f v1, blind offsets: base roll / pitch +-5 deg, full yaw, base velocity +-0.3 m/s and +-0.5 rad/s, joints
                   +-0.15 rad (8.6 deg = the stance difference between the two policies) / +-1 rad/s, tilt termination 0.8
  gene_bank        lm7f v2, the handover itself: 30 % of the resetting envs start from a sample of the WALK policy's recorded stance
                   (assets/pose_banks/walk_lm7f_kit_footbonus6_still_2026-10-26.pt: zero-command stands 1-9 s after walking under
                   0.3 m/s pushes; q, qd, height, roll / pitch, base velocity + 0.03 rad / 0.3 rad/s blind noise), tilt termination
                   0.8. capture_spawn_state is re-appended AFTER the bank term: the anti-drift rewards must measure from the bank
                   pose, not from the trunk's draw (event terms run in attribute order).

No reward term changes (Bible pre-flight: the levers are reset / termination terms). The rows apply the p16c recipe in the job
(set_param x2, applied before edit_raw) and one gene (edit_raw)."""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from unitree_rl_lab.tasks.locomotion import mdp

BANK = "walk_lm7f_kit_footbonus6_still_2026-10-26.pt"   # aspired-isaac-lab/assets/pose_banks/ (scripts/tools/record_pose_bank.py --walk_steps 150)


def _trunk_reset_asserts(cfg):
    rb = cfg.events.reset_base.params
    assert tuple(rb["pose_range"]["x"]) == (-0.1, 0.1) and tuple(rb["pose_range"]["yaw"]) == (-0.5, 0.5) and tuple(rb["velocity_range"]["x"]) == (0.0, 0.0), "[p16d] the balance trunk's still reset underneath"
    assert cfg.events.reset_robot_joints.func is mdp.reset_joints_by_scale, "[p16d] the trunk's reset_joints_by_scale underneath"
    assert getattr(cfg.terminations, "bad_orientation", None) is None, "[p16d] no tilt termination underneath"


def _tilt_termination(cfg):
    cfg.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


def _move_last(section, name):
    """re-append an existing term so it runs AFTER every term added before this call (managers iterate the cfg's __dict__ order)"""
    d = section.__dict__
    d[name] = d.pop(name)


def gene_resets(cfg):
    _trunk_reset_asserts(cfg)
    rb = cfg.events.reset_base.params["pose_range"]
    rb["x"] = (-0.5, 0.5); rb["y"] = (-0.5, 0.5); rb["yaw"] = (-3.14, 3.14)
    cfg.events.reset_robot_joints = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                                              params={"position_range": (-0.05, 0.05), "velocity_range": (-1.0, 1.0)})
    _tilt_termination(cfg)


def gene_activation(cfg):
    _trunk_reset_asserts(cfg)
    rb = cfg.events.reset_base.params
    rb["pose_range"]["roll"] = (-0.087, 0.087)
    rb["pose_range"]["pitch"] = (-0.087, 0.087)
    rb["pose_range"]["yaw"] = (-3.14, 3.14)
    for k in ("x", "y"):
        rb["velocity_range"][k] = (-0.3, 0.3)
    for k in ("roll", "pitch", "yaw"):
        rb["velocity_range"][k] = (-0.5, 0.5)
    cfg.events.reset_robot_joints = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
                                              params={"position_range": (-0.15, 0.15), "velocity_range": (-1.0, 1.0)})
    _tilt_termination(cfg)


def gene_bank(cfg):
    _trunk_reset_asserts(cfg)
    assert getattr(cfg.events, "reset_pose_bank", None) is None and getattr(cfg.events, "capture_spawn_state", None) is not None, "[bank] no bank term underneath; capture_spawn_state present"
    cfg.events.reset_pose_bank = EventTerm(func=mdp.reset_from_pose_bank, mode="reset",
                                           params={"bank": BANK, "prob": 0.3, "pos_noise": 0.03, "vel_noise": 0.3})
    _move_last(cfg.events, "capture_spawn_state")
    order = [k for k, v in cfg.events.__dict__.items() if isinstance(v, EventTerm) and v.mode == "reset"]
    assert order.index("reset_robot_joints") < order.index("reset_pose_bank") < order.index("capture_spawn_state"), f"[bank] reset order {order}"
    _tilt_termination(cfg)
