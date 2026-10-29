"""p16e (2026-10-29): the HANDOVER walk -> balance, from the stances the walk policies actually STOP in.

Operator 2026-10-29: "We need handover for both policies to be trained. Both locomotion -> balance and balance -> locomotion ... Balance
needs to take into account that the legs can be in a variety of poses." p16d_sw_bank trained from ONE walk policy's settled stand
(footbonus6_still, 1-9 s after the stop) and widened the lock-step engagement shift (-6.8 cm vs the baseline -2.9: a flag, no eye
verdict yet). p16e's bank is the switch's real entry distribution: the stops of FIVE walk policies (lm7f gforce_resets / gforce_airtime /
footbonus6_still, lm7g trunk / still_alive10 -- narrow, wide, staggered and lunging stands), sampled from 0.2 s to 6 s after the stop,
kept only where MovementModule's switch gate would let the balance policy in (both feet > 80 N, |base rate| < 0.3 rad/s, tilt < 0.15 rad):
assets/pose_banks/walk_mix5_switchgate_2026-10-29.pt (scripts/tools/merge_pose_banks.py over record_pose_bank.py --switch_gate).

Recipe = p16d's (the stancewide recipe from p14g_hips_armfull @75600; set_param x2 in the job) + one gene; the p16d helpers keep the
reset order (bank after the trunk's joints, capture_spawn_state re-appended last) and the tilt termination.
  bank_mix     30 % of resets from the mix bank (p16d's share)
  bank_mix50   50 % (the walk side's lm7h share)
"""
from isaaclab.managers import EventTermCfg as EventTerm

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg_p16d import _move_last, _tilt_termination, _trunk_reset_asserts

BANK_MIX = "walk_mix5_switchgate_2026-10-29.pt"   # aspired-isaac-lab/assets/pose_banks/


def _bank(cfg, prob):
    _trunk_reset_asserts(cfg)
    E = cfg.events
    assert getattr(E, "reset_pose_bank", None) is None and getattr(E, "capture_spawn_state", None) is not None, "[p16e] no bank underneath; capture present"
    E.reset_pose_bank = EventTerm(func=mdp.reset_from_pose_bank, mode="reset",
                                  params={"bank": BANK_MIX, "prob": prob, "pos_noise": 0.03, "vel_noise": 0.3})
    _move_last(E, "capture_spawn_state")
    order = [k for k, v in E.__dict__.items() if isinstance(v, EventTerm) and v.mode == "reset"]
    assert order.index("reset_robot_joints") < order.index("reset_pose_bank") < order.index("capture_spawn_state"), f"[p16e] reset order {order}"
    _tilt_termination(cfg)


def gene_bank_mix(cfg): _bank(cfg, 0.3)
def gene_bank_mix50(cfg): _bank(cfg, 0.5)
