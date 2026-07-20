"""Desk-line balance task — Balance-Desk (the manipulation-at-a-desk policy).

The product use-case is manipulation + balancing AT A DESK, which does NOT
need 1.5 m/s shove recovery (the desk is right there — a big shove means
crashing into it). This is a SEPARATE policy line from the general (demo)
balance, sharing the SYM foundation but with a different world and reward
priority:

  * GENTLE curriculum — small impulse pushes (<=0.3 m/s) + light sustained
    (<=15 N) + a constant micropush rain (+-0.15 m/s). Enough disturbance to
    keep reactions crisp (disturbance-discipline law) without forcing steps.
  * ORIENT 6-DoF arms (orientation_mode=True) as the CORE competency —
    arms are the desk's primary activity, so full pose range matters most.
  * STAY-IN-PLACE — stronger symmetric base homing.
  * DESK ZONE — a dense one-sided penalty on forward drift into the desk in
    front (base_forward_zone_penalty). NOT a termination: a shove toward the
    desk is not an episode-ending failure, and terminating on unavoidable
    contact teaches nothing. Falls stay the only termination (base_height).

Same queue-override mechanics as Balance-QIK (QUEUE_JOB_JSON applied LAST, so
jobs win). Warmstart from a general SYM winner to inherit recovery competence,
then this world refines it toward planted desk stillness.

Tunables (first-pass defaults, expect to sweep): push ceiling 0.30 m/s,
sustained 15 N, desk-zone threshold 0.10 m / weight -3.0, base homing -1.5.
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .balance_env_cfg_queue_ik import _swap_in_ik_command


def _make_desk(cfg):
    """Apply the desk-line world + reward priority on top of the IK envelope."""
    # --- arms: IK envelope + 6-DoF orientation (the desk's core competency) ---
    _swap_in_ik_command(cfg)
    cfg.commands.arm_pose_command.orientation_mode = True

    # --- gentle curriculum: small pushes, no big shoves ---
    # impulse ceiling 0.30 m/s (was 1.5) — models a human bump, not a shove.
    cfg.curriculum.push_velocity.params["levels"] = (0.10, 0.15, 0.20, 0.25, 0.30)
    # sustained ceiling 15 N (was 50) — a hand resting / arm-reach reaction.
    cfg.curriculum.sustained_push.params["levels"] = (
        ((0.0, 0.0), (0.0, 0.0)),     # warmup
        ((0.0, 5.0), (1.5, 3.0)),
        ((0.0, 8.0), (1.5, 3.0)),
        ((0.0, 12.0), (2.0, 3.5)),
        ((0.0, 15.0), (2.0, 3.5)),
    )
    # constant micropush rain — keeps reactions crisp without forcing steps.
    cfg.events.micro_push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}},
    )

    # --- stay-in-place: stronger symmetric base homing (was -0.75) ---
    cfg.rewards.base_pos_xy_l2_from_spawn.weight = -1.5

    # --- desk zone: dense one-sided forward-drift penalty (NOT a termination) ---
    cfg.rewards.base_forward_zone = RewTerm(
        func=mdp.base_forward_zone_penalty,
        weight=-3.0,
        params={"threshold": 0.10},
    )
    # NB: feet_air stays -3 (per-touchdown); feet_slide stays -1. NOT amped —
    # amping feet_air would tilt the policy toward SLIDING over stepping.
    # Terminations: falls only (base_height) — unchanged.


@configclass
class RobotEnvCfgDesk(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_desk(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgDesk(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_desk(self)
        # play at full arm workspace immediately
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
