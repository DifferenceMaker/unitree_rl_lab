"""lm7k (2026-11-04): the stand on a floor that does not give + the walking torso + the start ramp + the arm sway, on lm7j_stack_clear10 @140175.

Robot 2026-11-04 (five balance->walk switches recorded at 500 Hz, then one with the arm park): the deploy chain is faithful (an offline
ONNX replay of the robot's own state reproduces rt/lowcmd to 0.00 rad; gains, obs, encoder table, 50 Hz cadence all match) and with the
arms parked the switch itself is clean -- and the walk STILL leans. Standing under the walk the real hip rolls pushed the pinned feet
sideways at 41-54 Nm each (out in one run, in the next), the ankles at 18-26 Nm, and the body settled 4.4 deg forward (another run 2.2
back) and never corrected; the rig stands the same policy at 1-25 Nm and 0-1 deg, p14h on the same robot at 0-6 Nm and +0.2..+0.7 deg.
In sim the same lateral push MOVES the feet (every lm7j row slides out during the exam's stand; the operator's "walks with its legs
inside and then out"), so the policy never learned to stand on feet that do not give. Operator 2026-11-04: "FF stays. Ramp2 stays.
stack_clear10 seems like the parent ... we need to attack the torso stability front and the over stability of the robot."

Genes (scripts/queue/lm7k/HYPOTHESIS.md; every row = stack_clear10's gene chain re-applied + exactly one lever, except legs_squeeze):
  legs      lm7j gene 6 (pod-killed, never tested): feet_width_hold_standing (std .03, w3, the width of the last stop), feet_width_walking
            (.30, std .06, w1.5), feet_yaw_hold_standing (std .1, w2) -- anchor the feet where they stopped instead of pushing them
  squeeze   + lateral_torque_standing = mdp.lateral_torque_l2_standing (hip roll / hip yaw / ankle roll, both legs, (tau/100)^2 at a zero
            command) x -3: the robot's 41+54 Nm = -1.4/s, the rig's 25 Nm = -0.19/s, 10 Nm = -0.03/s (a price on the push itself)
  legs_squeeze  both (the only stack; rule 36: read against the singles)
  slide     feet_slide -2 -> -8: the foot's planar speed while loaded, x4 -- the sim must not let the stance creep
  upright6  upright_bonus (upright_bonus_standing std .05) 3 -> 6: the operator's hypothesis as a control -- does a doubled standing
            upright income change the robot's lean (the data say the lean is a force equilibrium, not a choice)
  torso10   torso_stability_ang std_ang .15 -> .10 (weight 3): the walking torso front, the kernel-width lever once more
  roll30    torso_roll_walk -15 -> -30: the shoulder line, the price lever doubled
  ramp2     slew_acc inf -> (0.5, 0.5, 1.0): lm7j_ramp2's start ramp on top of clear10's stop ramp ("Ramp2 stays")
  armsway   lm7j gene 5 (pod-killed): arm_motion_walking x -0.25 ("the arms swing left to right. That needs to be fixed asap")
  straight  + foot_lateral_vel_walking = mdp.foot_lateral_vel_l2_walking (each foot's base-frame LATERAL velocity squared, summed, while
            walking) x -1.0: the swing kernel pays tanh(2 |v_xy|) for any planar foot speed, so the outward arc ("legs inside and then out",
            "dancing") earns like a straight swing; this prices its sideways part
  micropush + micro_push = push_by_setting_velocity every 1-3 s at +-0.15 m/s (the balance line's p13g/p14 recipe, on top of the walk's
            +-0.3 m/s every 8-12 s): the operator's question "how is this not a micropush solution" as a row. The balance line found the
            micro-push dither cost torso stillness (dp8 sway); here it is a CONTROL against squeeze / legs: does a policy that is
            disturbed every 1-3 s learn to correct a lean it would otherwise accept. p14h_stancewide HAS this event (events.micro_push,
            +-0.15 m/s every 1-3 s); the walk line never had it.
  micro_legs_squeeze  micropush + legs + squeeze: the disturbance axis composed with the stand fix (rule 36: read against the three singles)
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.locomotion import mdp

from .lm7j_env_cfg import stack_clear10, gene_legs, gene_armsway, _FEET

PARENT, IT = "lm7j_stack_clear10_2026-11-03", 140175   # the harvest pick (Exam v3b 64.7; robot 2026-11-04: the stand leans, see above)
SQUEEZE_W = -3.0      # one hip roll at 50 Nm = (0.5)^2 x 3 = 0.75/s; the robot's two hips (41 + 54 Nm) = -1.4/s; the rig's 25 Nm = -0.19/s
SLIDE_W = -8.0        # feet_slide -2 -> -8
UPRIGHT_W = 6.0       # upright_bonus 3 -> 6
TORSO_STD = 0.10      # torso_stability_ang std_ang .15 -> .10
ROLL30_W = -30.0      # torso_roll_walk -15 -> -30
ACC2 = (0.5, 0.5, 1.0)   # lm7j_ramp2's acceleration limit (vx, vy m/s^2; wz rad/s^2)
STRAIGHT_W = -1.0        # foot lateral velocity squared, summed over both feet, while walking (sized by the 90-it liveness)
MICRO_V, MICRO_S = 0.15, (1.0, 3.0)   # the balance line's micro_push: +-0.15 m/s velocity kicks every 1-3 s


def c10(cfg):
    """The parent's gene chain, bit-identical (JOB_VALIDITY 3): combo + ffarms + roll + stopramp2 + clear10."""
    stack_clear10(cfg)


def gene_squeeze(cfg):
    R = cfg.rewards
    assert getattr(R, "lateral_torque_standing", None) is None, "[squeeze] no lateral torque price underneath"
    R.lateral_torque_standing = RewTerm(func=mdp.lateral_torque_l2_standing, weight=SQUEEZE_W,
                                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(mdp.LATERAL_JOINTS)),
                                                "command_name": "base_velocity", "cmd_thr": 0.03, "scale_nm": 100.0})


def gene_slide(cfg):
    s = cfg.rewards.feet_slide
    assert s.weight == -2.0, "[slide] the parent's feet_slide -2"
    s.weight = SLIDE_W


def gene_upright6(cfg):
    u = cfg.rewards.upright_bonus
    assert u.func.__name__ == "upright_bonus_standing" and u.weight == 3.0 and u.params["std"] == 0.05, "[upright6] the parent's standing upright kernel std .05 w3"
    u.weight = UPRIGHT_W


def gene_torso10(cfg):
    t = cfg.rewards.torso_stability_ang
    assert t.func is mdp.torso_stability_bonus and t.weight == 3.0 and t.params["std_ang"] == 0.15, "[torso10] the parent's torso_stability_ang std .15 w3 (combo's torso15)"
    t.params["std_ang"] = TORSO_STD


def gene_roll30(cfg):
    r = cfg.rewards.torso_roll_walk
    assert r.func is mdp.torso_roll_rate_l2_walking and r.weight == -15.0, "[roll30] the parent's torso_roll_walk -15 (gene roll)"
    r.weight = ROLL30_W


def gene_acc(cfg):
    C = cfg.commands.base_velocity
    assert C.class_type is mdp.SlewVelocityCommand and tuple(C.slew_dec) == (1.0, 1.0, 2.0) and all(a == float("inf") for a in C.slew_acc), \
        "[ramp2] the parent's SlewVelocityCommand (dec 1/1/2 + the slew_dec curriculum, acc instant)"
    C.slew_acc = ACC2


def gene_straight(cfg):
    R = cfg.rewards
    assert getattr(R, "foot_lateral_vel_walking", None) is None, "[straight] no lateral foot-velocity price underneath"
    R.foot_lateral_vel_walking = RewTerm(func=mdp.foot_lateral_vel_l2_walking, weight=STRAIGHT_W,
                                         params={"asset_cfg": _FEET, "command_name": "base_velocity", "cmd_thr": 0.03})


def gene_micropush(cfg):
    E = cfg.events
    assert getattr(E, "micro_push", None) is None and E.push_robot.func is mdp.push_by_setting_velocity, "[micropush] the parent's push_robot (interval 8-12 s) and no micro_push"
    E.micro_push = EventTerm(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=MICRO_S,
                             params={"velocity_range": {"x": (-MICRO_V, MICRO_V), "y": (-MICRO_V, MICRO_V)}})


def lm7k_legs(cfg): c10(cfg); gene_legs(cfg)
def lm7k_squeeze(cfg): c10(cfg); gene_squeeze(cfg)
def lm7k_legs_squeeze(cfg): c10(cfg); gene_legs(cfg); gene_squeeze(cfg)
def lm7k_slide(cfg): c10(cfg); gene_slide(cfg)
def lm7k_upright6(cfg): c10(cfg); gene_upright6(cfg)
def lm7k_torso10(cfg): c10(cfg); gene_torso10(cfg)
def lm7k_roll30(cfg): c10(cfg); gene_roll30(cfg)
def lm7k_ramp2(cfg): c10(cfg); gene_acc(cfg)
def lm7k_armsway(cfg): c10(cfg); gene_armsway(cfg)
def lm7k_straight(cfg): c10(cfg); gene_straight(cfg)
def lm7k_micropush(cfg): c10(cfg); gene_micropush(cfg)
def lm7k_micro_legs_squeeze(cfg): c10(cfg); gene_micropush(cfg); gene_legs(cfg); gene_squeeze(cfg)   # the micro-push axis composed with the stand fix (operator 2026-11-04)
