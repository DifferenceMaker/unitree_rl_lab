"""Balance-Desk5b — dp5 with the moving-desk machinery REMOVED and the
illegal arm targets fixed.

WHY A SECOND TRUNK. The dp5 batch was 8-for-8 detonations. The autopsy found
two mistakes stacked on top of each other, and both of them are ours:

  1. HALF OF EVERY DESK DRAW WAS A POINT INSIDE THE TABLE. dp5 drew targets
     at `desk_height_w 1.00 +- desk_z_jitter 0.05`, i.e. uniform [0.95, 1.05],
     against a slab occupying exactly [0.95, 1.00]. The IK cannot resolve a
     point inside a kinematic solid, so the arm answers the residual the only
     way it can: it presses. desk_hit scaled dead-linearly with the draw rate
     (~4.1 per unit desk_level_prob: 0.0 -> -0.587, 0.3 -> -1.95, 0.6 -> -2.5),
     which is the signature of a command the policy can never satisfy rather
     than a skill it has not learned yet.

  2. THE DESK TELEPORTED INTO THE ROBOT. place_desk_follow re-placed the slab
     every 1-2 s to track an anchor that itself jumped every 1-2 s. A kinematic
     body relocated inside a limb resolves as an impulse, and the anchor
     observation moved with no cause the policy could feel. Both are unfair in
     the strict sense: no policy can predict either from its observations.

WHAT REPLACES THE MOVING ANCHOR. Nothing, deliberately. The mechanic that was
wanted -- the robot ends up displaced relative to the table and has to work
from there -- is exactly what the push ladder already produces, except with
real physics and a cause the robot can feel through its own IMU. A 1.0 m/s
shove IS anchor motion, expressed in the only frame the robot actually has.
So: static slab, static anchor, ladder untouched at 1.0 m/s / 30 N.

DELTAS vs Balance-Desk5 (everything else inherited unchanged):
  * desk draws lifted ABOVE the slab   1.00 +- 0.05  ->  1.10 +- 0.03
    (worst case 1.07, clear of the measured p90 hand-contact band -- see the
    DESK_DRAW_Z note: the commanded point is the WRIST, the hand hangs below it)
  * desk_slab_clip ON                  no target of ANY kind (free/start/desk)
    may sit inside or under the footprint -- this is the -0.587 floor that
    dp4c_deskcol2 paid with desk-level draws switched entirely off
  * move_anchor            REMOVED
  * anchor_repoint         REMOVED
  * place_desk_follow      REMOVED     (place_desk at reset stays: the slab is
                                        still placed relative to each env's own
                                        spawn, it just never moves again)

DELIBERATELY UNCHANGED: the push ladder (1.0 m/s / 30 N, 20-30 s / 30-50 s),
anchor_hold 8.0 sigma 0.15 heading_to="spawn", cheap stepping, the gait clock,
desk_hit -1.0 on the wrists. A hand that finds itself under the table and has
to work out what to do is a REAL situation and stays in the task; what leaves
is us ordering it there.
"""
from isaaclab.utils import configclass

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_desk import _make_desk
from .balance_env_cfg_desk5 import _make_desk5, DESK_TOP_Z
from .balance_env_cfg_queue import _apply_overrides, _load_overrides

# Draw plane. NOT desk_top + epsilon: the IK commands the wrist_yaw_link
# ORIGIN, but the hand is MERGED into that link (URDF import log: "L_hand_base_link
# has body properties and is being merged into left_wrist_yaw_link"), so the
# collision mesh reaches well below the commanded point. Measured on 108k
# contact samples (dp5_base policy, Desk5b): desk contact fires with the wrist
# origin at median 0.999, p90 1.062, max 1.187 -- i.e. the hand reaches ~6 cm
# below the origin in 90% of poses and up to 19 cm with fingers extended.
# 1.10 puts the wrist 10 cm over the slab, which is also simply where a wrist
# IS when the hand is reaching for an object sitting ON a table.
DESK_DRAW_Z = 1.10
DESK_DRAW_JITTER = 0.03   # -> [1.07, 1.13], clear of the p90 contact band


def _make_desk5b(cfg):
    """dp5b deltas on top of the dp5 trunk."""
    # --- 1. legal arm targets ---
    c = cfg.commands.arm_pose_command
    c.desk_height_w = DESK_DRAW_Z
    c.desk_z_jitter = DESK_DRAW_JITTER
    c.desk_slab_clip = True
    c.desk_slab_top = DESK_TOP_Z          # the PHYSICAL top (1.0), not the draw plane
    c.desk_slab_size = (0.5, 2.0)         # must match scene.desk spawn size
    c.desk_clear_z = 0.07                 # clip floor = the draw plane's own floor

    # --- 2. the desk and the anchor both stop moving ---
    # None-ing a term is how the managers drop it; the slab is still PLACED at
    # reset (per-env spawn yaw), it just never gets relocated afterwards.
    cfg.events.move_anchor = None
    cfg.events.anchor_repoint = None
    cfg.events.place_desk_follow = None


@configclass
class RobotEnvCfgDesk5b(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_desk(self)
        _make_desk5(self)
        _make_desk5b(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgDesk5b(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_desk(self)
        _make_desk5(self)
        _make_desk5b(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
