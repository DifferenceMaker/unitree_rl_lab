"""lm7h (2026-10-29): the HANDOVER (balance -> walk, from any arm pose) and the ZERO-COMMAND STANCE (stand like p14h_stancewide).

Hardware 2026-10-29 (both greenlit lm7f walks): stepping backwards / inward legs right after the switch from p14h_stancewide, stumbles
at a zero command. Operator: "We need handover for both policies to be trained ... Locomotion needs to take into account that the
[arms] can be up or in manipulation". Evening sim2sim of lm7g (scripts/queue/lm7g/CONCLUSIONS.md): MAIN OBJECTIVE = "p14h_stancewide
balance perfect standing for locomotion at 0 commanded velocity"; two parents, "We already have 2 possible parents":
  A = lm7g_still_alive10_2026-10-28 @112396 ("Stance somewhat better ... keeps still"; carries the resets gene) -- the ladder
  B = lm7g_airtime_clear9_2026-10-28 @112396 ("Really good stance ... stancewide perfect. STill also"; no resets) -- the decisive rows
Friction DR is in EVERY row ("permanently add fric to the soup without it being a separate trunk. Make is always there").
(The first authoring on lm7f_kit_gforce_resets smoked 8/8 + lineage PASS the same evening and was re-targeted.)

Genes (mdp/lm7h.py for the terms; every reward gate is the COMMAND):
  bank_sw     50 % of resets start from p14h_stancewide's own recorded stand (assets/pose_banks/balance_p14h_stancewide_armdeploy_2026-10-29.pt:
              1024 envs x 12 s under 0.3 m/s pushes, 30 % of them holding the REAL stack's go_to_start arms -- right arm up, left arm at
              the side -- the rest across the trained arm envelope; feet 0.325 m apart) + 0.03 rad / 0.3 rad/s noise, then a ZERO
              command for 3 s (the switch only fires at a zero command), capture_spawn_state re-appended after the bank (the stand-hold
              terms measure from the handover pose), tilt termination 0.8 (already the parent's).
  stance_p14h the p14h stance at a zero command: foot_stance_tracking nominal +-0.16 (measured; walk +-0.10 under a command),
              feet_too_near .28 (walk .22), feet_too_far .45 (walk .60)
  no_step     p14h's step price at a zero command: -1 x 0.4 per touchdown
  sway_still  torso stillness kernels at a zero command: lin std .06 m/s, ang std .21 rad/s, +1.5 each -- sized by Bible rule 31 from the
              90-it liveness smoke (frozen actor = the parent's behaviour in Isaac, pushes on): first draft .04 / .09 earned 13 % / 0.5 % of
              the ceiling (the MuJoCo rig's 0.063 rad/s rms underestimated Isaac's pushed stand); widened so the parent earns ~37 %
  fric        lm7g's friction DR U[.8, 1.2] -- the soup (every row)
  pc          the privileged feet critic on parent A: padded warmstart lm7g_still_alive10_pc10pad_2026-10-29, critic warm-up 1000, the
              ladder held for those 1000 iterations
  energy / pivot   lm7g's energy_gated (-0.01, full at a zero command) / feet_pivot (-2.0) as axes (operator: energy_gated "fric + this +
              H6? Axis run it"; pivot "This one definitely stays")
Rows on A: H1 bank_sw | H2 +stance_p14h | H3 +no_step | H4 +sway_still | H6 H4+pc | H6e H6+energy_gated | H4p H4+pivot | C1 stance_p14h | C2 no_step
Rows on B: H3 | H4 (the parent A/B at the decisive rows). Every row = parent + fric + its genes.
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.locomotion import mdp

from .lm7e_env_cfg import _FEET_SENSOR
from .lm7g_env_cfg import gene_ctl_clear9, gene_energy_gated, gene_fric, gene_pivot, gene_trunk_still_alive10, parent_ga, parent_gr
from .lm7e_env_cfg import gene_pc_contacts
from isaaclab.managers import TerminationTermCfg as DoneTerm
from .lm7u_env_cfg import _FEET

BANK_SW = "balance_p14h_stancewide_armdeploy_2026-10-29.pt"   # aspired-isaac-lab/assets/pose_banks/
STAND_NOMINAL = [[0.01, 0.16], [0.01, -0.16]]                 # p14h_stancewide's measured stand (bank feet_b_mean +0.159 / -0.167)
THR = 0.03                                                    # the walk line's standing gate
PARENT_A, PARENT_B, IT = "lm7g_still_alive10_2026-10-28", "lm7g_airtime_clear9_2026-10-28", 112396
PC10PAD_A = "lm7g_still_alive10_pc10pad_2026-10-29"           # PARENT_A with the critic's first layer padded 120 -> 130 (lm7e/pad_critic.py)


# ── the parents (re-applied so every row = the parent milestone's env + exactly its genes) ──
def parent_a(cfg): parent_gr(cfg); gene_trunk_still_alive10(cfg)
def parent_b(cfg): parent_ga(cfg); gene_ctl_clear9(cfg)


def _move_last(section, name):
    d = section.__dict__
    d[name] = d.pop(name)


def gene_bank_sw(cfg):
    E = cfg.events
    assert getattr(E, "reset_pose_bank", None) is None and getattr(E, "capture_spawn_state", None) is not None, "[bank_sw] no bank underneath; capture present"
    bo = getattr(cfg.terminations, "bad_orientation", None)
    if bo is None:                                            # parent B (no resets gene): the bank rows' tilt termination (lm7f bank)
        cfg.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    assert cfg.terminations.bad_orientation.params["limit_angle"] == 0.8, "[bank_sw] tilt termination 0.8"
    E.reset_pose_bank = EventTerm(func=mdp.reset_from_pose_bank_hold, mode="reset",
                                  params={"bank": BANK_SW, "prob": 0.5, "pos_noise": 0.03, "vel_noise": 0.3, "hold_s": 3.0})
    _move_last(E, "capture_spawn_state")
    order = [k for k, v in E.__dict__.items() if isinstance(v, EventTerm) and v.mode == "reset"]
    assert order.index("reset_robot_joints") < order.index("reset_pose_bank") < order.index("capture_spawn_state"), f"[bank_sw] reset order {order}"
    E.handover_command_hold = EventTerm(func=mdp.handover_command_hold, mode="interval", interval_range_s=(0.0, 0.0),
                                        is_global_time=True, params={"command_name": "base_velocity"})


def gene_stance_p14h(cfg):
    R = cfg.rewards
    t = R.foot_stance_tracking
    assert t.func is mdp.foot_stance_tracking and t.weight == 1.5 and t.params["std"] == 0.08, "[stance_p14h] the parent's stance kernel 1.5 @ .08"
    assert [list(x) for x in t.params["nominal_foot_pos_b"]] == [[0.0, 0.1], [0.0, -0.1]], "[stance_p14h] from the walk nominal +-0.10"
    t.func = mdp.foot_stance_tracking_split
    t.params = {"asset_cfg": t.params["asset_cfg"], "std": 0.08, "nominal_stand": STAND_NOMINAL, "nominal_walk": [[0.0, 0.10], [0.0, -0.10]],
                "command_name": "base_velocity", "cmd_thr": THR}
    n, f = R.feet_too_near, R.feet_too_far
    assert n.weight == -8.0 and n.params["threshold"] == 0.22 and f.weight == -8.0 and f.params["threshold"] == 0.6, "[stance_p14h] hinges .22 / .60 @ -8"
    n.func = mdp.feet_too_near_split
    n.params = {"asset_cfg": n.params["asset_cfg"], "threshold_stand": 0.28, "threshold_walk": 0.22, "command_name": "base_velocity", "cmd_thr": THR}
    f.func = mdp.feet_too_far_split
    f.params = {"asset_cfg": f.params["asset_cfg"], "threshold_stand": 0.45, "threshold_walk": 0.6, "command_name": "base_velocity", "cmd_thr": THR}


def gene_no_step(cfg):
    assert getattr(cfg.rewards, "feet_step_price", None) is None, "[no_step] no step price underneath"
    cfg.rewards.feet_step_price = RewTerm(func=mdp.feet_step_price_standing, weight=-1.0,
                                          params={"sensor_cfg": _FEET_SENSOR, "touchdown_penalty": 0.4, "command_name": "base_velocity", "cmd_thr": THR})


def gene_sway_still(cfg):
    R = cfg.rewards
    assert getattr(R, "torso_still_lin", None) is None and getattr(R, "torso_still_ang", None) is None, "[sway_still] none underneath"
    torso = SceneEntityCfg("robot", body_names=["torso_link"])
    R.torso_still_lin = RewTerm(func=mdp.torso_still_lin_standing, weight=1.5, params={"std": 0.06, "command_name": "base_velocity", "cmd_thr": THR, "asset_cfg": torso})
    R.torso_still_ang = RewTerm(func=mdp.torso_still_ang_standing, weight=1.5, params={"std": 0.21, "command_name": "base_velocity", "cmd_thr": THR, "asset_cfg": torso})


def gene_pc_a(cfg):
    """lm7g's pc recipe on parent A: the privileged feet critic + the tracking ladder HELD for the 1000 warm-up iterations, anchored at
    parent A's iteration (lm7g.gene_pc anchors at lm7f's 105397)."""
    gene_pc_contacts(cfg)
    t = cfg.curriculum.track_std
    assert t.params.get("hold_until_step", 0) == 0, "[pc] no hold underneath"
    t.params["hold_until_step"] = (IT + 1000) * 24


# ── rows: every row = parent + fric (the soup) + its genes ──
def _h1(cfg): gene_fric(cfg); gene_bank_sw(cfg)
def _h2(cfg): _h1(cfg); gene_stance_p14h(cfg)
def _h3(cfg): _h2(cfg); gene_no_step(cfg)
def _h4(cfg): _h3(cfg); gene_sway_still(cfg)
def a_h1(cfg): parent_a(cfg); _h1(cfg)
def a_h2(cfg): parent_a(cfg); _h2(cfg)
def a_h3(cfg): parent_a(cfg); _h3(cfg)
def a_h4(cfg): parent_a(cfg); _h4(cfg)
def a_h6(cfg): parent_a(cfg); _h4(cfg); gene_pc_a(cfg)
def a_h6e(cfg): a_h6(cfg); gene_energy_gated(cfg)
def a_h4p(cfg): parent_a(cfg); _h4(cfg); gene_pivot(cfg)
def a_c1(cfg): parent_a(cfg); gene_fric(cfg); gene_stance_p14h(cfg)
def a_c2(cfg): parent_a(cfg); gene_fric(cfg); gene_no_step(cfg)
def b_h3(cfg): parent_b(cfg); _h3(cfg)
def b_h4(cfg): parent_b(cfg); _h4(cfg)
