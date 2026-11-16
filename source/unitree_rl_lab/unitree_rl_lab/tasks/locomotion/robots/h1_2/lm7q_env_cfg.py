"""lm7q (2026-11-16): the encoder ladder CONTINUED on t57 -- +-0.010 -> 0.012 -> 0.015 rad (authored + smoked; NOT launched).

Operator 2026-11-16: "Bigger offset dr? ... maybe instead of 0.01 we go 0.015?" / "Make it rung-like. From 0.01 to 0.012 to 0.015 or
something like that. Just a continuation of what we had and for 5k iters or so."

Parent: t57 = milestone s3_t57_2026-11-15 @169648 (the s3 study's best feasible point). It trains with the band HELD at +-0.010 rad on
the 13 leg + torso joints (study_s2's gene_enc(S2_BAND): events.encoder_offset, policy obs in the encoder frame, target - offset, critic
true) and carries NO encoder ladder. The measured robot residual is larger than that band in the pitch chain (ankle pitch 0.0305 rad,
hip roll 0.020, ankle roll 0.013: study_env_cfg.RESID), and a fixed +-0.03 band broke training (p16_dr enc03), so the band grows by rungs.

Row (aspired-isaac-lab scripts/queue/lm7q/HYPOTHESIS.md):
  enc_up   study_env_cfg.study_s4(cfg) with every lever at its default (= t57's env exactly) + ONE lever: curriculum.enc_ladder =
           mdp.encoder_offset_ladder (the lm7m function, unchanged) with rungs (0.010, 0.012, 0.015) rad on t57's own event (same 13
           joints, same uniform draw per joint per reset). The first rung IS t57's band, so iteration 0 is t57's world and the env.yaml
           differs from the milestone's in curriculum.enc_ladder only. Promotion = the lm7p rule: at a 60 s episode boundary, when
           >= 90 % of the window's ended episodes timed out (no fall) AND the no-decay guard holds (the mean squared vx error of the
           window's timed-out episodes <= 1.25 x the first eligible window's), after >= 300 iterations on the rung and never before
           parent_it + 400 (the 300-iteration critic warm-up + 100). Never back, never two rungs in one check.
           The no-decay term is track_vx here: study_s2's gene_err_split replaced track_err (the lm7n-lm7p guard's term) by the
           per-axis prices; track_vx's raw value is (vx_cmd - vx)^2, independent of the tracking-std ladder like track_err's was.

Timing (STEPS_PER_IT 24, episode 3000 steps = 125 it): the resumed counter is 169648 x 24 = 4071552, the boundaries fall at +102,
+227, +352, +477, +602, +727, +852 ...; hold_until = +400 -> the first eligible check is +477 (0.012 at the earliest), the 300-it hold
rounds up to 375 -> 0.015 at +852 at the earliest, which leaves ~4150 of 5000 iterations on the top rung.
"""
from isaaclab.managers import CurriculumTermCfg as CurrTerm

from unitree_rl_lab.tasks.locomotion import mdp

from .lm7l_env_cfg import SW_ENC01_JOINTS
from .lm7m_env_cfg import ENC_SURVIVE, ENC_HOLD_ITERS, ENC_WARMUP_ITERS, STEPS_PER_IT
from .lm7n_env_cfg import NODECAY_TOL
from .study_env_cfg import study_s4, S4_PARENT_MILESTONE, S4_PARENT_IT, S2_BAND

PARENT, IT = "s3_t57_2026-11-15", 169648
ENC_LADDER_UP = (0.010, 0.012, 0.015)   # rad: 0.57 / 0.69 / 0.86 deg per joint per episode, uniform; the first rung = t57's held band
NODECAY_TERM_UP = "track_vx"            # t57 has no track_err (gene_err_split): the vx price carries the no-decay guard

assert (PARENT, IT) == (S4_PARENT_MILESTONE, S4_PARENT_IT) and ENC_LADDER_UP[0] == S2_BAND, "[lm7q] study_s4's parent = t57 @169648; the first rung = its band"


def gene_enc_ladder_up(cfg, levels=ENC_LADDER_UP, parent_it=IT, nodecay_term=NODECAY_TERM_UP):
    """The ladder on a cfg that ALREADY carries the band (lm7m / lm7n's genes call gene_enc first and need a clean cfg): adds only the
    curriculum term. Loud when the band underneath is not the first rung on the 13 joints (then iteration 0 is not the parent's
    world), when a ladder is already there, when the rungs do not climb, or when the no-decay term is not a live price."""
    ev = getattr(cfg.events, "encoder_offset", None)
    assert ev is not None and ev.func is mdp.randomize_encoder_offset and ev.mode == "reset" and "offset_range_per_joint" not in ev.params \
        and tuple(ev.params["offset_range"]) == (-levels[0], levels[0]), f"[enc_ladder_up] the parent's band = +-{levels[0]} (one band, reset event) underneath"
    assert list(ev.params["asset_cfg"].joint_names) == list(SW_ENC01_JOINTS), "[enc_ladder_up] the parent's 13 leg + torso joints underneath"
    assert cfg.observations.policy.joint_pos_rel.func is mdp.joint_pos_rel_encoder, "[enc_ladder_up] the policy reads the encoder frame"
    assert getattr(cfg.curriculum, "enc_ladder", None) is None, "[enc_ladder_up] no ladder underneath"
    assert len(levels) >= 2 and all(b > a > 0 for a, b in zip(levels, levels[1:])), f"[enc_ladder_up] rungs climb, got {levels}"
    if nodecay_term is not None:
        t = getattr(cfg.rewards, nodecay_term, None)
        assert t is not None and t.weight < 0, f"[enc_ladder_up] rewards.{nodecay_term} = a live price (the no-decay guard divides by its weight)"
    cfg.curriculum.enc_ladder = CurrTerm(func=mdp.encoder_offset_ladder,
                                         params={"event_term_name": "encoder_offset", "levels": tuple(levels), "survive_frac": ENC_SURVIVE,
                                                 "gate_term": "track_lin_vel_xy", "gate_frac": None,
                                                 "min_hold_steps": ENC_HOLD_ITERS * STEPS_PER_IT,
                                                 "hold_until_step": (parent_it + ENC_WARMUP_ITERS) * STEPS_PER_IT,
                                                 "nodecay_term": nodecay_term, "nodecay_tol": NODECAY_TOL})


def lm7q_enc_up(cfg): study_s4(cfg); gene_enc_ladder_up(cfg)
