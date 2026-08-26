"""Desk6 — the LEAN COMMAND trunk (colleague's design, operator-ratified
2026-08-21). Operator: "He said he will publish the lean he needs and the
desk-line policy takes it in. My immediate reaction was to make a reward for
tracking that lean."

The contract: a 1-dim commanded pelvis PITCH in RADIANS (gravity-referenced,
forward-positive), sampled U[-0.10, +0.35] with 30% zero-snaps in training;
at deploy the colleague's publisher feeds the same obs slot. +1 obs dim =
OBS CONTRACT BREAK vs every dp5 policy — Desk6 is SCRATCH-ONLY by design
(operator ruling: scratch first, transplant_handless-style surgery as the
fallback if scratch loses the table manners).

Economy = Desk5b trunk + the dp5e_pain VALIDATED deltas baked (trunk-is-task:
pain is the first no-lean desk policy — hardware candidate; its economy is
the Desk6 floor) + the lean-tracking pair swapped in for the upright pair:
  * upright_bonus (sigma 0.01 proj-gravity — would fight any commanded lean)
    -> lean_track_bonus (kernel on (pitch-cmd)^2 + roll^2, sigma 0.05 rad;
    at cmd=0 this IS an upright bonus)
  * flat_orientation_l2 -> flat_orientation_lean_l2 (roll always priced,
    pitch priced against the COMMAND — the walk flat-split pattern ported)
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp.commands import LeanCommandCfg

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .balance_env_cfg_desk5b import _make_desk, _make_desk5, _make_desk5b


def _make_desk6(cfg):
    from isaaclab.managers import ObservationTermCfg as ObsTerm

    # ---- the lean command channel ----
    cfg.commands.lean_command = LeanCommandCfg(
        ranges=(-0.10, 0.35), zero_prob=0.3, resampling_time_range=(4.0, 8.0))
    cfg.observations.policy.lean_command = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "lean_command"})
    cfg.observations.critic.lean_command = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "lean_command"})

    # ---- the lean-tracking pair (replaces the upright pair) ----
    cfg.rewards.upright_bonus = RewTerm(
        func=mdp.lean_track_bonus, weight=1.0,
        params={"std": 0.05, "command_name": "lean_command"})
    cfg.rewards.flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_lean_l2, weight=-2.5,
        params={"command_name": "lean_command"})

    # ---- dp5e_pain validated deltas, baked (its overrides.json 1:1) ----
    c = cfg.commands.arm_pose_command
    c.desk_level_prob = 0.60
    c.cycle_mode = True
    c.cycle_pattern = ("start", "desk", "desk")
    c.cycle_random_prob = 0.25
    c.start_pose_b = (0.250, 0.490, 0.550)
    c.default_pose_prob = 0.10
    cfg.rewards.desk_hit.weight = 0.0
    cfg.rewards.joint_deviation_hips.weight = -2.5
    cfg.rewards.anchor_hold.weight = 12.0
    cfg.rewards.undesired_contacts.weight = -100.0
    cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces",
        body_names=["pelvis", "torso_link", ".*hip.*", ".*knee.*"])
    cfg.rewards.desk_reach = RewTerm(
        func=mdp.desk_reach_bonus, weight=8.0, params={"sigma": 0.25})


@configclass
class RobotEnvCfgDesk6(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_desk(self)
        _make_desk5(self)
        _make_desk5b(self)
        _make_desk6(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgDesk6(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_desk(self)
        _make_desk5(self)
        _make_desk5b(self)
        _make_desk6(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())


# ---------------------------------------------------------------------------
# Desk6b (dp6b, operator 2026-08-26): Desk6 + the desk_penetration TERMINATION.
# dp6_leancmd previews: "almost half the robots get pushed inside the table";
# the -100 undesired_contacts fired only -1.15/ep because a body through the
# slab collider reports no contact. Weight-free geometric termination (pelvis /
# torso_link centre inside the slab footprint below the desk top): the
# impossible state is cut, the unavoidable shove is not punished. Same obs
# contract as Desk6 (91) -> warmstart from dp6_leancmd_resume. "No need to
# overengineer beyond that."
# ---------------------------------------------------------------------------
def _make_desk6b(cfg):
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    cfg.terminations.desk_penetration = DoneTerm(
        func=mdp.desk_penetration,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "torso_link"]),
                "desk_name": "desk", "xy_margin": 0.0, "z_margin": 0.02})


@configclass
class RobotEnvCfgDesk6b(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_desk(self)
        _make_desk5(self)
        _make_desk5b(self)
        _make_desk6(self)
        _make_desk6b(self)
        _apply_overrides(self, _load_overrides())


@configclass
class RobotPlayEnvCfgDesk6b(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_desk(self)
        _make_desk5(self)
        _make_desk5b(self)
        _make_desk6(self)
        _make_desk6b(self)
        _apply_overrides(self, _load_overrides())
