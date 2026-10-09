"""WorkLean / WorkLean88 -- p14h_stancewide with a lean command, at the desk (dp9d, operator 2026-10-09: "My idea is that
the p14h_stancewide is a good policy and i want that policy but just with leaning. Is that so hard to get? Can't we just
take its terms and make some of them lean-gated or something like that? As one axis run?").

THE LEDGER = p14h_stancewide's, every term at its value (copied from _p14h_stancewide_ref(), built the way its run was
built, not transcribed), changed ONLY where leaning needs it or the desk forbids something:
  lean  upright_bonus         upright_bonus 3.0 / std .05 (keep vertical) -> lean_track_bonus 8.0 / std .10 on
                              lean_command (the Desk8c tracker, the setting that brought the lean back in dp9c)
        flat_orientation_l2   flat_orientation_l2 -2.5 -> flat_orientation_lean_l2 -2.5 (pitch priced against the command)
        joint_deviation_hips  joint_deviation_l1 -1.0 -> joint_deviation_l1_lean_gated -1.0, lean_ref .35 (the hinge fade
                              at p14h's weight, not the desk's -2.5)
  desk  undesired_contacts    -0.75 on torso/hips/knees -> -100 on the Desk8b set incl. pelvis, SHOULDERS, ELBOWS (dp9:
                              every row ended lying on the desk, carried by the unpriced shoulder meshes)
        desk_hit              (absent) -> -1.0, wrists vs the slab, hinge 1 N, clamp 10 (Desk8b: discouraged, not forbidden)
Everything else is p14h's as it stands: energy -0.01, desk_reach 4.0 (require_desk_draw False), stance tracking 1.5 / .08 /
+-.15, feet_too_near .28, feet_clearance 5.0 always on, gait speed-gated, plain capture point and feet_air, feet_slide -1,
damp -3, homing -0.75, heading 1.5 / .10, torso split 3+3, the walls and caps, base_height 0 (nothing prices the height:
p14h's crouch is inherited). Not in the ledger, because p14h never had them: anchor_hold, base_forward_zone, the anchor
gating, hiproll_over_cap, foot_impact_force.

THE WORLD = the Desk8d world, unchanged (the slab and its wrist contact sensor, desk_penetration, the desk arm envelope:
desk draws / cycle mode / slab clip, lean command U[-.10,.35] with 30 % zero, the desk push regime 1.0 m/s / 30 N at
20-30 s / 30-50 s, micro_push), plus ONE change: contact_forces history 3 -> 4 as p14h (the Kitchen trunk set it; the
feet_air / gait / clearance terms read that history, and they are p14h's terms now).

WorkLean (93 obs): the desk contract, so dp9b_work_stancewide (93) warmstarts it. The anchor_point / anchor_yaw inputs stay
as UNREWARDED leftovers; they have been constants in training since Desk8 froze the anchor ("The policies ignore it").
WorkLean88 (88 obs): the anchor inputs removed from policy AND critic, leaving p14h's 87 + lean_command LAST (critic 117 + 1):
the layout milestones/p14h_stancewide_lean88pad_2026-10-09 was padded for (transplant_v3_to_v4 --extra_obs 1).

GENES: gene_p14h_curric (p14h's push curriculum and intervals: 8-12 s to 1.5 m/s, 15-25 s to 50 N) and gene_crouch (the
lean-gated height price at p14h_stancewide's measured standing height).
"""
import copy

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_desk8b import DESK8B_CONTACT_BODIES
from .balance_env_cfg_desk8d import _desk8d_stack, _p14h_stancewide_ref
from .balance_env_cfg_queue import _apply_overrides, _load_overrides

WORKLEAN_LEAN_W = 8.0
WORKLEAN_LEAN_STD = 0.10
WORKLEAN_HINGE_REF = 0.35
WORKLEAN_CONTACT_W = -100.0
WORKLEAN_DESK_HIT_W = -1.0
WORKLEAN_CONTACT_HISTORY = 4
# the 4 ledger rows WorkLean changes (desk_hit is added); every other row must equal p14h_stancewide's
WORKLEAN_CHANGED = ("upright_bonus", "flat_orientation_l2", "joint_deviation_hips", "undesired_contacts")
# gene_crouch target, MEASURED 2026-10-09 (scripts/queue/dp9d/probe_stand.py, 64 envs, pushes off, 5-25 s):
# p14h_stancewide model_82475 stands at pelvis z 0.989 +- 0.004 (knee 0.617 rad, default 0.60, 0 falls);
# dp9b_work_stancewide model_90025 at lean 0 stands at 1.026 +- 0.009 on near-straight knees (0.115 rad)
WORKLEAN_CROUCH_TARGET = 0.99
WORKLEAN_CROUCH_W = -15.0


def _terms(grp):
    return [n for n, t in grp.__dict__.items() if isinstance(t, ObsTerm)]


def _same_rew(a, b):
    return a.func is b.func and a.weight == b.weight and a.params == b.params


def _make_worklean(cfg):
    ref = _p14h_stancewide_ref().rewards
    cfg.rewards = copy.deepcopy(ref)
    R = cfg.rewards
    # 1. the lean channel
    R.upright_bonus = RewTerm(func=mdp.lean_track_bonus, weight=WORKLEAN_LEAN_W,
                              params={"std": WORKLEAN_LEAN_STD, "command_name": "lean_command"})
    R.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_lean_l2, weight=R.flat_orientation_l2.weight,
                                    params={"command_name": "lean_command"})
    R.joint_deviation_hips.func = mdp.joint_deviation_l1_lean_gated
    R.joint_deviation_hips.params["command_name"] = "lean_command"
    R.joint_deviation_hips.params["lean_ref"] = WORKLEAN_HINGE_REF
    # 2. desk safety
    R.undesired_contacts.weight = WORKLEAN_CONTACT_W
    R.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=DESK8B_CONTACT_BODIES)
    R.desk_hit = RewTerm(func=mdp.desk_hit_penalty, weight=WORKLEAN_DESK_HIT_W, params={"force_thr": 1.0, "max_val": 10.0})
    # 3. the one world change
    cfg.scene.contact_forces.history_length = WORKLEAN_CONTACT_HISTORY

    # asserts: p14h's ledger + exactly the changes above
    names = {n for n, t in R.__dict__.items() if isinstance(t, RewTerm)}
    ref_names = {n for n, t in ref.__dict__.items() if isinstance(t, RewTerm)}
    assert names == ref_names | {"desk_hit"}, f"[WorkLean] term set = p14h's + desk_hit, got +{names - ref_names} -{ref_names - names}"
    drift = [n for n in ref_names if n not in WORKLEAN_CHANGED and not _same_rew(getattr(R, n), getattr(ref, n))]
    assert not drift, f"[WorkLean] rows drifted from p14h_stancewide: {drift}"
    assert R.upright_bonus.func is mdp.lean_track_bonus and R.upright_bonus.weight == 8.0 and R.upright_bonus.params == {"std": 0.10, "command_name": "lean_command"}, "[WorkLean] lean tracker 8 / .10"
    assert R.flat_orientation_l2.func is mdp.flat_orientation_lean_l2 and R.flat_orientation_l2.weight == -2.5, "[WorkLean] lean-aware L2 at p14h's -2.5"
    assert R.joint_deviation_hips.func is mdp.joint_deviation_l1_lean_gated and R.joint_deviation_hips.weight == -1.0 and R.joint_deviation_hips.params["lean_ref"] == 0.35, "[WorkLean] hinge fade at p14h's -1.0"
    assert R.undesired_contacts.weight == -100.0 and list(R.undesired_contacts.params["sensor_cfg"].body_names) == DESK8B_CONTACT_BODIES, "[WorkLean] desk contact set -100"
    assert R.energy.weight == -0.01 and R.desk_reach.weight == 4.0 and R.desk_reach.params.get("require_desk_draw") is False and R.base_height.weight == 0.0, "[WorkLean] p14h's energy / reach / no height price"
    assert cfg.scene.desk is not None and cfg.terminations.desk_penetration is not None and cfg.commands.lean_command is not None, "[WorkLean] desk world"
    assert tuple(cfg.curriculum.push_velocity.params["levels"])[-1] == 1.0 and tuple(cfg.events.push_robot.interval_range_s) == (20.0, 30.0), "[WorkLean] desk push regime"
    assert cfg.commands.arm_pose_command.desk_level_prob == 0.6 and cfg.commands.arm_pose_command.desk_slab_clip is True, "[WorkLean] desk arm envelope"


def _make_worklean88(cfg):
    """The anchor inputs out of policy and critic: p14h's observation groups + lean_command last."""
    for grp in (cfg.observations.policy, cfg.observations.critic):
        grp.anchor_point = None
        grp.anchor_yaw = None
    ref = _p14h_stancewide_ref().observations
    for gname in ("policy", "critic"):
        g, r = getattr(cfg.observations, gname), getattr(ref, gname)
        assert _terms(g) == _terms(r) + ["lean_command"], f"[WorkLean88] {gname} = p14h's + lean_command last, got {_terms(g)}"
        bad = [n for n in _terms(r) if getattr(g, n) != getattr(r, n)]
        assert not bad, f"[WorkLean88] {gname} terms differ from p14h's: {bad}"


def gene_p14h_curric(cfg):
    """dp9d_p14h_curric (operator 2026-10-09: "make one run where the curric is also the p14h_stancewide's"): the BALANCE
    push regime -- push_robot every 8-12 s to 1.5 m/s, sustained every 15-25 s to 50 N -- instead of the desk's."""
    ref = _p14h_stancewide_ref()
    cfg.curriculum.push_velocity = copy.deepcopy(ref.curriculum.push_velocity)
    cfg.curriculum.sustained_push = copy.deepcopy(ref.curriculum.sustained_push)
    cfg.events.push_robot.interval_range_s = ref.events.push_robot.interval_range_s
    cfg.events.sustained_push_apply.interval_range_s = ref.events.sustained_push_apply.interval_range_s
    assert tuple(cfg.curriculum.push_velocity.params["levels"])[-1] == 1.5 and tuple(cfg.events.push_robot.interval_range_s) == (8.0, 12.0), "[p14h_curric] 1.5 m/s every 8-12 s"
    assert tuple(cfg.curriculum.sustained_push.params["levels"])[-1][0][1] == 50.0 and tuple(cfg.events.sustained_push_apply.interval_range_s) == (15.0, 25.0), "[p14h_curric] 50 N every 15-25 s"


def gene_crouch(cfg):
    """dp9d_p14h_crouch (operator 2026-10-09: "Can the base_height be set at the crouch with the weight but gated on the lean
    command?"): -15 x (z - 0.99)^2 x (1 - |lean cmd| / .35): full price standing upright, none while a lean is commanded."""
    cfg.rewards.base_height = RewTerm(func=mdp.base_height_l2_lean_gated, weight=WORKLEAN_CROUCH_W,
                                      params={"target_height": WORKLEAN_CROUCH_TARGET, "command_name": "lean_command",
                                              "lean_ref": WORKLEAN_HINGE_REF})
    assert cfg.rewards.base_height.func is mdp.base_height_l2_lean_gated and cfg.rewards.base_height.weight == -15.0, "[crouch] gated height -15"
    assert cfg.rewards.base_height.params["target_height"] == 0.99 and cfg.rewards.base_height.params["lean_ref"] == 0.35, "[crouch] target .99, fade over .35"


@configclass
class RobotEnvCfgWorkLean(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _desk8d_stack(self)
        _make_worklean(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgWorkLean(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _desk8d_stack(self)
        _make_worklean(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())


@configclass
class RobotEnvCfgWorkLean88(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _desk8d_stack(self)
        _make_worklean(self)
        _make_worklean88(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgWorkLean88(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _desk8d_stack(self)
        _make_worklean(self)
        _make_worklean88(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
