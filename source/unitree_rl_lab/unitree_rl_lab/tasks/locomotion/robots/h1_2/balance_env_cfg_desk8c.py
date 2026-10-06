"""Desk8c-Work -- Desk8b-Work with the lean kernel re-opened (2026-10-06).

WHAT dp9b SHOWED (four transplant rows on Desk8b-Work, sim2sim 2026-10-06 on the desk rig, --hand790):
  "Sways around. Rotates. No leaning." (work) / "straight legs ... Barely any leaning. None. IT is stable
  though ... Doesn't try to go back to the anchor." (stancewide) / "Also straight legs. Sways left to
  right. No leaning." (explore) / "Some bent legs. Sways left to right ... The feet are rotating inward?"
  (balw). The lying-on-the-desk pose is GONE (the Desk8b fixes held: desk penetration 0.2-0.8 %,
  undesired_contacts ~0 with the shoulders in the set). What is left is the WORK list: leaning,
  stillness, a crouch, recoveries without gaiting back.

THE LEAN DIAGNOSIS (lineage read, 2026-10-06): the lean tracker `upright_bonus` (mdp.lean_track_bonus,
w 8) earned 82-87 % of its weight in every policy that LEANED -- dp6c / dp7_hinge / dp8 / dp8b -- and
all of them ran it at std 0.10. dp8c tightened it to 0.035 ("the snap tracker") and still earned 74 %
because it was warmstarted from a policy that already leaned. The dp9b parents (p14g / p14h) have
never leaned: at std 0.035 a 20-degree error is 10 sigma out, the kernel is FLAT there, and the four
dp9b picks earn 1.8-2.1 of 8 = exactly the zero-command share (zero_prob 0.3 x 8 = 2.4) -- the income
of a policy that never follows the command. Operator: "Yeah detighten the kernel."

THE CROUCH (operator: "I do prefer the crouch but not at the expense of leaning"): base_height
(base_height_lean_l2 -15, target 0.87 cos(lean)) was the lever that over-seated dp6c ("the one run where
it was too much seating": the hinge drops the pelvis beyond the cosine target and the term paid -0.24/step
for it) -- Desk7 zeroed it for that reason and Desk8 brought it back. Here it goes back to 0: the crouch
lever that does not fight the hinge geometry is the knee anchor (joint_deviation_knees), an AXIS row.

THE DELTA vs Desk8b-Work (everything else identical, see balance_env_cfg_desk8b.py):
  upright_bonus.std   0.035 -> 0.10   (the std every leaning policy learned under)
  base_height         -15.0 -> 0.0    (Desk7's setting; the dp6c over-seat lever off)
Parent: milestones/dp9b_work_stancewide_2026-10-05 (model_90025, the sim2sim pick: "stancewide was
better"); procedure = the stack + the std reset (the only dp9b row that ended healthy).
"""
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_desk8b import _desk8b_stack
from .balance_env_cfg_queue import _apply_overrides, _load_overrides

DESK8C_LEAN_STD = 0.10
DESK8C_BASE_HEIGHT_W = 0.0


def _make_desk8c_work(cfg):
    R = cfg.rewards
    # 1. the lean kernel back at the std the lean was learned under (dp6c / dp7 / dp8 / dp8b)
    R.upright_bonus.params["std"] = DESK8C_LEAN_STD
    # 2. the over-seat lever off (the knee anchor is the crouch axis, not the pelvis height)
    R.base_height.weight = DESK8C_BASE_HEIGHT_W

    assert R.upright_bonus.func is mdp.lean_track_bonus and R.upright_bonus.weight == 8.0 and R.upright_bonus.params["std"] == 0.10 and R.upright_bonus.params["command_name"] == "lean_command", "[Desk8c-Work] lean tracker 8 / 0.10"
    assert R.base_height.weight == 0.0 and R.base_height.func is mdp.base_height_lean_l2, "[Desk8c-Work] base_height off (term kept for the ledger)"
    assert R.flat_orientation_l2.weight == -2.5 and R.joint_deviation_knees.weight == -0.4, "[Desk8c-Work] lean L2 partner -2.5 and knee anchor -0.4 stay trunk values (axis rows move them)"
    assert R.desk_hit.weight == -1.0 and ".*shoulder.*" in R.undesired_contacts.params["sensor_cfg"].body_names and tuple(cfg.curriculum.push_velocity.params["levels"])[-1] == 1.0, "[Desk8c-Work] Desk8b fixes intact underneath"
    assert cfg.events.micro_push is not None and R.torso_stability_lin.params["std_lin"] == 0.15, "[Desk8c-Work] micro_push on and torso lin std .15 in the trunk (axis rows move them)"


def _desk8c_stack(cfg):
    _desk8b_stack(cfg)
    _make_desk8c_work(cfg)


@configclass
class RobotEnvCfgDesk8cWork(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _desk8c_stack(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgDesk8cWork(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _desk8c_stack(self)
        self.commands.arm_pose_command.debug_vis = True
        self.curriculum.ik_workspace.params["warmup_steps"] = 0
        self.curriculum.ik_workspace.params["scale_levels"] = (1.0,)
        _apply_overrides(self, _load_overrides())
