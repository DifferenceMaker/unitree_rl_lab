"""Desk8b-Work -- Desk8-Work + the two fixes the dp9 harvest demanded (2026-10-05).

WHAT dp9 SHOWED (four scratch rows on Desk8-Work, 20k it, previews + an Isaac contact probe):
  1. Every row ends up LYING ON THE DESK. The pose is carried by the SHOULDER meshes, the elbows
     and the hands -- none of which is in `undesired_contacts` (the old balance rule "NO
     shoulder/elbow -- arms are not policy-controlled") -- while `torso_link` reads 0 N in 100 %
     of chest-on-slab steps (its collider is an 8 x 16 x 10 cm box mid-torso; the real torso is
     20 x 32 x 76 cm; the shoulder meshes hold the box a few cm above the slab). Probe,
     dp9_work_balw: chest over the slab 69 % of env-steps, torso 0 N, shoulders 142-676 N, hands up
     to 1.8 kN. dp9_work (-100): counted contacts 0.003/step (the price is obeyed on what it
     sees) but hands on the desk at 1.7 kN and the pelvis lowered to 0.8 m. The desk IS inside
     the contact sensor (hips/pelvis against the slab edge ARE counted); the support bodies are
     simply unpriced. `desk_hit` (hand-vs-desk, dp4c) was zeroed in Desk6 and dropped in Desk7.
  2. Desk penetration ended 14-54 % of episodes: the BALANCE push regime (8-12 s to 1.5 m/s,
     15-25 s to 50 N) borrowed on 09-30 drives the pelvis into a slab 0.5 m ahead.

OPERATOR RULINGS (2026-10-05): shoulders and elbows may NOT rest on the table (hard price);
hands MAY touch it but must not be encouraged (soft, clamped price); the desk line keeps the
DESK push regime -- 1.0 m/s and 30 N "as with all Desk line policies" -- while the general
balance line keeps 1.5 m/s; the tougher balance policy comes in by TRANSPLANT (87 -> 93 obs).

THE DELTA vs Desk8-Work (everything else identical, see balance_env_cfg_desk8.py):
  undesired_contacts  body set + .*shoulder.* + .*elbow.*  (weight stays -100)
  desk_hit            RE-ADDED at -1.0: desk-side contact sensor filtered to the two wrist_yaw
                      links, hinge above 1 N, CLAMPED at 10 -> at most -10/step for leaning on
                      the hands (vs alive +30): discouraged, not forbidden
  push regime         DESK: push_robot 20-30 s, levels 0.25..1.0 m/s; sustained 30-50 s,
                      0 -> 10 -> 20 -> 30 N (Desk5 values, carried by Desk7)
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_desk8 import _desk8_stack
from .balance_env_cfg_queue import _apply_overrides, _load_overrides

DESK8B_CONTACT_BODIES = ["pelvis", "torso_link", ".*hip.*", ".*knee.*", ".*shoulder.*", ".*elbow.*"]
DESK8B_DESK_HIT_W = -1.0
DESK8B_PUSH_LEVELS = (0.25, 0.5, 0.75, 1.0)
DESK8B_SUSTAINED_LEVELS = (
    ((0.0, 0.0), (0.0, 0.0)),
    ((0.0, 10.0), (1.5, 3.0)),
    ((0.0, 20.0), (2.0, 3.5)),
    ((0.0, 30.0), (2.0, 4.0)),
)
DESK8B_PUSH_INTERVAL = (20.0, 30.0)
DESK8B_SUSTAINED_INTERVAL = (30.0, 50.0)


def _make_desk8b_work(cfg):
    R = cfg.rewards
    # 1. the support bodies become visible to the hard price
    R.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=DESK8B_CONTACT_BODIES)
    # 2. hands on the desk: discouraged (hinge above 1 N, clamped at 10 N-equivalent), not forbidden
    R.desk_hit = RewTerm(func=mdp.desk_hit_penalty, weight=DESK8B_DESK_HIT_W,
                         params={"force_thr": 1.0, "max_val": 10.0})
    # 3. the DESK push regime (the slab is 0.5 m ahead)
    cfg.events.push_robot.interval_range_s = DESK8B_PUSH_INTERVAL
    cfg.events.sustained_push_apply.interval_range_s = DESK8B_SUSTAINED_INTERVAL
    cfg.curriculum.push_velocity.params["levels"] = DESK8B_PUSH_LEVELS
    cfg.curriculum.sustained_push.params["levels"] = DESK8B_SUSTAINED_LEVELS

    # asserts: the delta is what the docstring says, and Desk8-Work underneath is intact
    bn = R.undesired_contacts.params["sensor_cfg"].body_names
    assert ".*shoulder.*" in bn and ".*elbow.*" in bn and "torso_link" in bn and R.undesired_contacts.weight == -100.0, "[Desk8b-Work] shoulders/elbows priced at -100"
    assert R.desk_hit.weight == DESK8B_DESK_HIT_W and R.desk_hit.func is mdp.desk_hit_penalty and R.desk_hit.params["max_val"] == 10.0, "[Desk8b-Work] desk_hit soft hand price"
    assert tuple(cfg.curriculum.push_velocity.params["levels"])[-1] == 1.0 and tuple(cfg.events.push_robot.interval_range_s) == (20.0, 30.0), "[Desk8b-Work] desk push regime 1.0 m/s"
    assert tuple(cfg.curriculum.sustained_push.params["levels"])[-1][0][1] == 30.0 and tuple(cfg.events.sustained_push_apply.interval_range_s) == (30.0, 50.0), "[Desk8b-Work] sustained to 30 N"
    assert cfg.events.seed_offset is None and cfg.events.move_anchor is None and R.base_height.weight == -15.0 and R.upright_bonus.params["std"] == 0.035, "[Desk8b-Work] Desk8-Work intact underneath"


def _desk8b_stack(cfg):
    _desk8_stack(cfg)
    _make_desk8b_work(cfg)


@configclass
class RobotEnvCfgDesk8bWork(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _desk8b_stack(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgDesk8bWork(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _desk8b_stack(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
