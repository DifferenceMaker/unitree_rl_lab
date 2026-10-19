"""Desk8d-Work -- Desk8c-Work + energy -0.01, the desk line's new default (dp9d, operator 2026-10-09: "From now on the
energy term is default. At the p14h_stancewide rate. energy term goes to dp line.").

WHY. The balance line has carried mdp.energy at -0.01 since the Kitchen trunk ("a dose that can be seen, about 1 % of
return"); Desk8-Work (09-30) left it out on purpose ("energy is an axis row, not baked") and every desk policy since --
dp9, dp9b, dp9c -- trained with nothing that makes fast loaded motion expensive. The 10-09 ledger comparison: action_rate
-7.7..-8.2 on the desk line vs -3.2 on p14g_hips_armfull / p14h_stancewide, action_cap -0.41 vs -0.07, knee_torque_cap
2-3x, with action_rate and action_cap configured IDENTICALLY on both lines (-0.6 / -0.05 thr 1.5 squared) -- pure
behaviour. dp9b_work_scratch ("so jiggery", sim2sim + its Isaac preview) is the extreme of it.

THE DELTA vs Desk8c-Work: energy -0.01 (the Kitchen term 1:1: sum |qvel| x |applied torque|, i.e. power -- a static
crouch is free under it, jitter is not). Nothing else.

Also here: `_p14h_stancewide_ref()` -- p14h_stancewide's own config (HipsArm trunk + its job's two set_params), BUILT
rather than transcribed, the single source of every term the dp9d rows take from p14h (gene_merge here, the WorkLean
ledger in balance_env_cfg_worklean.py) -- and `gene_merge`, the operator's merge list.
Parent: milestones/dp9b_work_stancewide_2026-10-05 (model_90025); procedure = the stack + the std reset (dp9c's).
"""
import copy

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_desk8c import _desk8c_stack
from .balance_env_cfg_hipsarm import _make_hipsarm
from .balance_env_cfg_hipsarm_dr import gene_phys, install_dr_hooks
from .balance_env_cfg_kitchen import _use_real_hand
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .balance_env_cfg_queue_ik import _swap_in_ik_command

DESK8D_ENERGY_W = -0.01

# p14h_stancewide = Unitree-H1_2-Balance-HipsArm-FIXLR + scripts/queue/p14h/p14h_stancewide.job (two set_params; the
# tuple-of-lists is what the queue's set_param makes of the job's JSON list, so the copy matches the milestone env.yaml)
P14H_FEET_TOO_NEAR = 0.28
P14H_NOMINAL_FOOT = ([0.0, 0.15], [0.0, -0.15])


def _p14h_stancewide_ref():
    """A fresh p14h_stancewide config, built exactly as its training run built it (HipsArm class order: IK command,
    hipsarm stack, real hand, then the job's set_params). Read-only reference: callers deep-copy what they take."""
    ref = RobotEnvCfg()
    _swap_in_ik_command(ref)
    _make_hipsarm(ref)
    _use_real_hand(ref)
    ref.rewards.feet_too_near.params["threshold"] = P14H_FEET_TOO_NEAR
    ref.rewards.foot_stance_tracking.params["nominal_foot_pos_b"] = P14H_NOMINAL_FOOT
    R = ref.rewards
    assert R.energy.weight == -0.01 and R.upright_bonus.func is mdp.upright_bonus and R.joint_deviation_hips.weight == -1.0, "[p14h ref] HipsArm ledger"
    assert R.foot_stance_tracking.weight == 1.5 and R.foot_stance_tracking.params["std"] == 0.08 and R.feet_clearance.weight == 5.0, "[p14h ref] stance/clearance as the milestone"
    return ref


def _make_desk8d_work(cfg):
    R = cfg.rewards
    R.energy = RewTerm(func=mdp.energy, weight=DESK8D_ENERGY_W, params={})

    assert R.energy.func is mdp.energy and R.energy.weight == -0.01, "[Desk8d-Work] energy -0.01 (the p14h_stancewide rate)"
    assert R.upright_bonus.func is mdp.lean_track_bonus and R.upright_bonus.params["std"] == 0.10 and R.base_height.weight == 0.0, "[Desk8d-Work] Desk8c underneath (lean tracker .10, base_height 0)"
    assert R.desk_hit.weight == -1.0 and ".*elbow.*" in R.undesired_contacts.params["sensor_cfg"].body_names and tuple(cfg.curriculum.push_velocity.params["levels"])[-1] == 1.0, "[Desk8d-Work] Desk8b fixes underneath"


def _desk8d_stack(cfg):
    _desk8c_stack(cfg)
    _make_desk8d_work(cfg)


def gene_merge(cfg):
    """dp9d_merge -- p14h_stancewide merged into the desk trunk, the operator's list 1:1 (2026-10-09: "One run is basically
    the p14h_stancewide merged into dp. The terms get overriden except for those that produce leaning"). Desk stays: the
    hinge (hips -2.5 lean-gated), the lean tracker (8 / .10), flat_orientation lean-aware -2.5, undesired_contacts -100 with
    shoulders/elbows, gait anchor-gated, anchor_hold, base_forward_zone, desk_hit, hiproll_over_cap, foot_impact_force, the
    desk curriculum. base_height comes BACK at the lean-aware -15 (Desk8/8b; Desk8c had zeroed it). desk_reach takes p14h's
    weight only (4.0); its desk-draw gate stays the desk line's."""
    ref = _p14h_stancewide_ref().rewards
    R = cfg.rewards
    R.feet_slide.weight = -1.0
    R.torso_lin_vel_xy.weight = -3.0
    R.base_pos_xy_l2_from_spawn.weight = -0.75
    R.base_height.weight = -15.0
    for name in ("foot_stance_tracking", "feet_too_near", "feet_clearance", "capture_point_touchdown", "feet_air_time_step"):
        setattr(R, name, copy.deepcopy(getattr(ref, name)))
    R.heading_stable_bonus.params["std"] = 0.10
    R.desk_reach.weight = 4.0

    assert R.energy.weight == -0.01 and R.feet_slide.weight == -1.0 and R.torso_lin_vel_xy.weight == -3.0 and R.base_pos_xy_l2_from_spawn.weight == -0.75, "[merge] energy / slide / damp / homing at p14h's"
    assert R.undesired_contacts.weight == -100.0 and ".*shoulder.*" in R.undesired_contacts.params["sensor_cfg"].body_names, "[merge] undesired_contacts stays -100 incl. shoulders"
    assert R.joint_deviation_hips.func is mdp.joint_deviation_l1_lean_gated and R.joint_deviation_hips.weight == -2.5, "[merge] the hinge stays desk"
    assert R.upright_bonus.func is mdp.lean_track_bonus and R.upright_bonus.weight == 8.0 and R.upright_bonus.params["std"] == 0.10, "[merge] lean tracker 8 / .10"
    assert R.flat_orientation_l2.func is mdp.flat_orientation_lean_l2 and R.flat_orientation_l2.weight == -2.5, "[merge] lean-aware L2 -2.5"
    assert R.base_height.func is mdp.base_height_lean_l2 and R.base_height.weight == -15.0 and R.base_height.params["target_height"] == 0.87, "[merge] lean-aware base_height -15 @ .87"
    assert R.foot_stance_tracking.weight == 1.5 and R.foot_stance_tracking.params["std"] == 0.08 and tuple(map(tuple, R.foot_stance_tracking.params["nominal_foot_pos_b"])) == ((0.0, 0.15), (0.0, -0.15)), "[merge] p14h stance tracking"
    assert R.feet_too_near.params["threshold"] == 0.28 and R.feet_clearance.func is mdp.foot_clearance_reward and R.feet_clearance.weight == 5.0 and R.feet_clearance.params["target_height"] == 0.15, "[merge] p14h walls + clearance"
    assert R.capture_point_touchdown.func is mdp.capture_point_touchdown_distance and "gate_dist" not in R.capture_point_touchdown.params and R.capture_point_touchdown.weight == -5.0, "[merge] capture point plain"
    assert R.feet_air_time_step.weight == -1.0 and "free_when_displaced_m" not in R.feet_air_time_step.params, "[merge] feet_air plain"
    assert R.gait.params.get("gate_anchor_dist") == 0.20 and R.anchor_hold.weight == 12.0 and R.base_forward_zone.weight == -6.0, "[merge] gait anchor-gated, anchor terms kept"
    assert R.heading_stable_bonus.weight == 1.5 and R.heading_stable_bonus.params["std"] == 0.10 and R.desk_reach.weight == 4.0, "[merge] heading std .10, desk_reach 4"



# ── dp9e genes (2026-10-12): one lever each on dp9d_merge @94375 (= Desk8d-Work + gene_merge), combos compose them in order ──
def gene_stance(cfg):
    """The short stance (merge already carries p14h's .28 / +-.15 and still stands short; hardware 10-12 tied the sway to a narrow
    stance): move the TARGET (p12 note), keep the std .08 kernel (p11 stance25: widening the kernel killed the diamond)."""
    R = cfg.rewards
    assert R.feet_too_near.params["threshold"] == 0.28 and tuple(map(tuple, R.foot_stance_tracking.params["nominal_foot_pos_b"])) == ((0.0, 0.15), (0.0, -0.15)), "[stance] from p14h's values"
    R.feet_too_near.params["threshold"] = 0.32
    R.foot_stance_tracking.params["nominal_foot_pos_b"] = ((0.0, 0.18), (0.0, -0.18))


def gene_still(cfg):
    """torso_stability_ang std_ang .30 -> .60: the kernel earns 8 % of 3 at the merge pick (Bible rule 31: widen a flat kernel from
    its measured fraction; the dp8b_still row that never ran)."""
    R = cfg.rewards
    assert R.torso_stability_ang.weight == 3.0 and R.torso_stability_ang.params["std_ang"] == 0.3, "[still] from the split's .30"
    R.torso_stability_ang.params["std_ang"] = 0.6


def gene_energy2(cfg):
    """energy -0.01 -> -0.02 (Bible rule 38: -0.01 is visible on hardware and not over-dosed) -- the 'unsure' recoveries."""
    assert cfg.rewards.energy.weight == -0.01, "[energy2] from the desk default"
    cfg.rewards.energy.weight = -0.02


def gene_calm(cfg):
    """micro_push every 1-3 s -> 3-6 s (dp7 sway diagnosis: the dither is trained; dp9c_nomicro removed it and detonated)."""
    ev = cfg.events.micro_push
    assert tuple(ev.interval_range_s) == (1.0, 3.0), "[calm] from Desk8's 1-3 s"
    ev.interval_range_s = (3.0, 6.0)


def gene_phys_desk(cfg):
    """p16_dr_phys's four physical bands (mass / friction / gains / lag) on the desk world: the DR hooks + gene_phys."""
    install_dr_hooks(cfg)
    gene_phys(cfg)


def gene_stance_still(cfg): gene_stance(cfg); gene_still(cfg)
def gene_stance_still_energy2(cfg): gene_stance_still(cfg); gene_energy2(cfg)
def gene_stance_still_energy2_calm(cfg): gene_stance_still_energy2(cfg); gene_calm(cfg)


# ── dp9f (2026-10-19): the anchor attractor as the sway suspect. Parent dp9e_stance_still_energy2 @99050 (operator: "by far the
# best"); every row re-applies merge + stance_still_energy2 (gene_dp9f_base) and changes ONE anchor knob (or calm). ──
def gene_dp9f_base(cfg):
    gene_merge(cfg)
    gene_stance_still_energy2(cfg)


def _anchor(cfg):
    A = cfg.rewards.anchor_hold
    assert A.func is mdp.anchor_hold_bonus and A.weight == 12.0 and A.params["deadband"] == 0.05 and A.params["sigma"] == 0.15, "[dp9f] from the merge's anchor_hold 12 / deadband .05 / sigma .15"
    return A


def gene_zone15(cfg):
    """anchor_hold deadband .05 -> .15 m: the flat zone, bigger (inside 5 cm the kernel pays 12/s in full, at 10 cm it has already
    lost 1.26/s -- the dominant position gradient by two orders of magnitude; the WORK line has no transit, stepping is locomotion's)."""
    _anchor(cfg).params["deadband"] = 0.15


def gene_zone25(cfg):
    _anchor(cfg).params["deadband"] = 0.25


def gene_sigma30(cfg):
    """sigma .15 -> .30 with the .05 band kept: the alternative shape, a soft pull instead of a flat zone."""
    _anchor(cfg).params["sigma"] = 0.30


def gene_noanchor(cfg):
    """anchor_hold 12 -> 0 on an otherwise UNCHANGED desk ledger (never run: the only anchor-free rows, WorkLean88, swapped the
    whole ledger too). Forward zone, heading terms and the anchor OBSERVATIONS stay (reward-only; no transplant)."""
    _anchor(cfg).weight = 0.0


def gene_zone15_calm(cfg): gene_zone15(cfg); gene_calm(cfg)
def gene_zone25_calm(cfg): gene_zone25(cfg); gene_calm(cfg)
def gene_noanchor_calm(cfg): gene_noanchor(cfg); gene_calm(cfg)


@configclass
class RobotEnvCfgDesk8dWork(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _desk8d_stack(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgDesk8dWork(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _desk8d_stack(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
