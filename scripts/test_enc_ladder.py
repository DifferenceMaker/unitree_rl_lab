"""Encoder-ladder promotion logic, unit level (no simulator): mdp/lm7m.py::encoder_offset_ladder driven by a stub env.

The REAL function is loaded from the repo file (lm7m.py imports only torch at run time); the env is a stub that carries exactly what
the function reads: common_step_counter, max_episode_length(_s), the event term's params, the reward manager's episode sums + weights
and the termination manager's time_outs. The survival signal and the no-decay signal are scripted per scenario.

Parameters = the lm7q row's (robots/h1_2/lm7q_env_cfg.py: rungs 0.010 / 0.012 / 0.015, survive 0.90, hold 300 it, none before +400,
no-decay on track_vx x 1.25), read from the cfg file's constants by ast so the test cannot drift from the gene.
Usage: python -I scripts/test_enc_ladder.py   -> "ALL PASS"
"""
import ast
import importlib.util
import io
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
LOCO = REPO / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion"
spec = importlib.util.spec_from_file_location("lm7m_ladder", LOCO / "mdp/lm7m.py")
lm7m = importlib.util.module_from_spec(spec); spec.loader.exec_module(lm7m)
ladder = lm7m.encoder_offset_ladder


def consts(path, names):
    """top-level NAME = <literal> (or A, B = <literals>) assignments of a cfg file, without importing it (it needs isaaclab)."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            tgt = node.targets[0]
            try:
                val = ast.literal_eval(node.value)
            except Exception:
                try:
                    val = eval(compile(ast.Expression(node.value), "<cfg>", "eval"), {}, dict(out))   # e.g. 300 + 100
                except Exception:
                    continue
            if isinstance(tgt, ast.Name):
                out[tgt.id] = val
            elif isinstance(tgt, ast.Tuple):
                out.update({t.id: v for t, v in zip(tgt.elts, val)})
    return {n: out[n] for n in names}


H12 = LOCO / "robots/h1_2"
Q = consts(H12 / "lm7q_env_cfg.py", ["IT", "ENC_LADDER_UP", "NODECAY_TERM_UP"])
M = consts(H12 / "lm7m_env_cfg.py", ["ENC_SURVIVE", "ENC_HOLD_ITERS", "ENC_WARMUP_ITERS", "STEPS_PER_IT", "ENC_LADDER"])
N = consts(H12 / "lm7n_env_cfg.py", ["NODECAY_TOL"])
IT, LEVELS, SPI = Q["IT"], Q["ENC_LADDER_UP"], M["STEPS_PER_IT"]
EP, EP_S, W_VX, NENV = 3000, 60.0, -1.4473, 8192
PARAMS = dict(event_term_name="encoder_offset", levels=LEVELS, survive_frac=M["ENC_SURVIVE"], gate_term="track_lin_vel_xy", gate_frac=None,
              min_hold_steps=M["ENC_HOLD_ITERS"] * SPI, hold_until_step=(IT + M["ENC_WARMUP_ITERS"]) * SPI,
              nodecay_term=Q["NODECAY_TERM_UP"], nodecay_tol=N["NODECAY_TOL"])
START = IT * SPI

bad = 0


def ok(c, m):
    global bad
    print(("OK  " if c else "BAD ") + m); bad += 0 if c else 1


class StubEnv:
    """what encoder_offset_ladder touches, nothing else."""
    def __init__(self, band, nodecay_term="track_vx", weight=W_VX):
        self.device = "cpu"
        self.common_step_counter = 0
        self.max_episode_length, self.max_episode_length_s = EP, EP_S
        self.event_cfg = SimpleNamespace(params={"offset_range": (-band, band)})
        self.event_manager = SimpleNamespace(get_term_cfg=lambda name: self.event_cfg)
        sums = {"track_lin_vel_xy": torch.zeros(NENV), nodecay_term: torch.zeros(NENV)}
        cfgs = {"track_lin_vel_xy": SimpleNamespace(weight=15.0), nodecay_term: SimpleNamespace(weight=weight)}
        self.reward_manager = SimpleNamespace(_episode_sums=sums, get_term_cfg=lambda name: cfgs[name])
        self.termination_manager = SimpleNamespace(time_outs=torch.zeros(NENV, dtype=torch.bool))


def run(survive, q_raw, iters=5000, params=PARAMS, band0=LEVELS[0], boundary_env=True, seed=0, quiet=True):
    """One resumed run. survive(it, level) -> the share of ended episodes that time out; q_raw(it, level) -> the mean squared vx error
    of an episode. Episode clocks are staggered like rsl_rl's random initial episode length: env i ends an episode whenever
    (step - phase_i) % 3000 == 0. boundary_env: env 0's clock sits ON the boundary steps (step % 3000 == 0), so every check has a
    caller; False removes every env whose clock sits there (the function is only called for resetting envs). Returns the per-
    iteration rung, the promotion iterations (relative), the event band history and the printed lines."""
    g = torch.Generator().manual_seed(seed)
    env = StubEnv(band0, nodecay_term=params["nodecay_term"] or "track_vx")
    phase = torch.randint(0, EP, (NENV,), generator=g)
    phase = torch.where(phase == 0, torch.ones_like(phase), phase)      # nobody on the boundary by chance ...
    if boundary_env:
        phase[0] = 0                                                      # ... except env 0 when asked
    order = torch.argsort(phase); sphase = phase[order]
    out = io.StringIO()
    rung, promos, bands = [], [], []
    with contextlib.redirect_stdout(out if quiet else sys.stdout):
        env.common_step_counter = START
        v = ladder(env, slice(None), **params)                            # train.py's resume path: compute(env_ids=None) -> a slice
        assert abs(float(v) - params["levels"][0]) < 1e-9
        level_now = 0
        for it in range(iters):
            for s in range(SPI):
                step = START + it * SPI + s + 1                           # the env increments the counter, then resets
                env.common_step_counter = step
                lo = torch.searchsorted(sphase, step % EP); hi = torch.searchsorted(sphase, step % EP, right=True)
                ids = order[lo:hi]
                if len(ids) == 0:
                    continue                                              # _reset_idx is only called with resetting envs
                p = survive(it, level_now)
                to = torch.rand(len(ids), generator=g) < p
                if len(ids) == 1 and p >= 0.5:
                    to[:] = True                                          # the lone boundary env follows the majority
                env.termination_manager.time_outs[ids] = to
                env.reward_manager._episode_sums[params["nodecay_term"] or "track_vx"][ids] = W_VX * q_raw(it, level_now) * EP_S
                env.reward_manager._episode_sums["track_lin_vel_xy"][ids] = 7.3 * EP_S
                v = float(ladder(env, ids, **params))
                k = min(range(len(params["levels"])), key=lambda i: abs(params["levels"][i] - v))
                if k != level_now:
                    promos.append((-(-(step - START) // SPI), level_now, k)); level_now = k
            rung.append(level_now); bands.append(env.event_cfg.params["offset_range"])
    ALL_RUNGS.append(rung)
    return rung, promos, bands, out.getvalue()


ALL_RUNGS = []
always = lambda it, lv: 1.0
flat_q = lambda it, lv: 0.020

# "+N" below = N iterations completed since the warmstart = (step - START) / 24, the convention of the waves' HYPOTHESIS files.
# ── 1. healthy robots: 0.010 -> 0.012 -> 0.015, one rung per check, at the predicted iterations, then it stays ──────────────────────
rung, promos, bands, log = run(always, flat_q)
b = [(s - START) / SPI for s in range(START + 1, START + 1000 * SPI) if s % EP == 0]         # the boundary steps, in iterations
ok(b[:4] == [102, 227, 352, 477], f"episode boundaries at +{[int(x) for x in b[:7]]} (resumed counter {START} = {IT} x {SPI})")
ok(rung[0] == 0 and bands[0] == (-0.010, 0.010), f"iteration 0: rung 0, the event band {bands[0]} = the parent's")
ok([p[1:] for p in promos] == [(0, 1), (1, 2)], f"climbs 0.010 -> 0.012 -> 0.015, one rung at a time: {[(f'+{i}', LEVELS[a], LEVELS[c]) for i, a, c in promos]}")
ok([p[0] for p in promos] == [477, 852], f"promotions at +{[p[0] for p in promos]} (first boundary after hold_until +400 = +477; then +375 = the 300-it hold rounded up to the 125-it boundary)")
ok(all(r == 0 for r in rung[:476]) and all(r == 1 for r in rung[476:851]) and all(r == 2 for r in rung[851:]), "the rung is 0.010 for 476 iterations, 0.012 until +852, 0.015 for the remaining 4148 of 5000")
ok(bands[475] == (-0.010, 0.010) and bands[476] == (-0.012, 0.012) and bands[851] == (-0.015, 0.015) and bands[-1] == (-0.015, 0.015),
   f"the reset event's offset_range follows the rung: after +476 {bands[475]}, after +477 {bands[476]}, after +852 {bands[851]}, after +5000 {bands[-1]}")
ok(log.count("PROMOTED") == 2 and "HELD" not in log.split("PROMOTED to level 2")[1], "after the top rung: no promotion, no HELD line (nothing above it)")
ok("no-decay baseline track_vx q = 0.02000" in log, "no-decay baseline = the first eligible window's q: " + [l for l in log.splitlines() if "baseline" in l][0].split(": ", 1)[1])
print("    " + "\n    ".join(l for l in log.splitlines() if "PROMOTED" in l))

# ── 2. no promotion during the critic warm-up, whatever the survival ────────────────────────────────────────────────────────────────
ok(min(p[0] for p in promos) >= M["ENC_WARMUP_ITERS"], f"no promotion before +{M['ENC_WARMUP_ITERS']} (critic warm-up 300 + 100): first at +{promos[0][0]}")

# ── 3. robots that fall: the rung is HELD, and it climbs when they survive again ────────────────────────────────────────────────────
rung, promos, bands, log = run(lambda it, lv: 0.85, flat_q, iters=2000)
ok(not promos and set(rung) == {0} and log.count("HELD") >= 10, f"survival 0.85 < 0.90: never promoted in 2000 it, {log.count('HELD')} HELD lines, band stays {bands[-1]}")
rung, promos, bands, log = run(lambda it, lv: 0.85 if it < 1100 else 1.0, flat_q, iters=2600)
ok([p[1:] for p in promos] == [(0, 1), (1, 2)] and promos[0][0] > 1100, f"survival recovers at +1100: promotions at +{[p[0] for p in promos]} (a window = the episodes since the last check)")

# ── 4. never back: survival collapses on the second rung -> held there, the band does not shrink ───────────────────────────────────
rung, promos, bands, log = run(lambda it, lv: 1.0 if lv == 0 else 0.30, flat_q, iters=3000)
ok([p[1:] for p in promos] == [(0, 1)] and all(r == 1 for r in rung[476:]) and bands[-1] == (-0.012, 0.012), "survival 0.30 on rung 0.012: HELD at 0.012, never demoted to 0.010, never promoted")

# ── 5. the no-decay guard: tracking decays on the wider band -> held; inside the tolerance -> promoted ─────────────────────────────
rung, promos, bands, log = run(always, lambda it, lv: 0.020 if lv == 0 else 0.0262, iters=2500)
ok([p[1:] for p in promos] == [(0, 1)] and rung[-1] == 1, "vx error x1.31 on rung 0.012 (> 1.25 x the baseline): HELD at 0.012 although survival is 1.0")
rung, promos, bands, log = run(always, lambda it, lv: 0.020 if lv == 0 else 0.0245, iters=2500)
ok([p[1:] for p in promos] == [(0, 1), (1, 2)], "vx error x1.225 (<= 1.25 x): promoted to 0.015")

# ── 6. a boundary step where no env resets = no check (the function is only called for resetting envs): the promotion slips ────────
rung, promos, bands, log = run(always, flat_q, iters=1500, boundary_env=False)
ok(not promos, "no env on the boundary clock: no check, no promotion (uniform clocks: P(empty boundary) = exp(-N/3000) = 0.065 per check at 8192 envs, 0.26 at 4096; a miss costs 125 it)")

# ── 7. the older waves' rungs through the same function (lm7n_enc_nodecay's parameters, parent 153199): the logged schedule ────────
P_OLD = dict(PARAMS, levels=M["ENC_LADDER"], nodecay_term="track_err", hold_until_step=(153199 + M["ENC_WARMUP_ITERS"]) * SPI)
START = 153199 * SPI
rung, promos, bands, log = run(always, flat_q, iters=1400, params=P_OLD, band0=M["ENC_LADDER"][0])
ok([p[0] for p in promos] == [426, 801, 1176] and bands[-1] == (-0.010, 0.010), f"lm7n's ladder (0.003 .. 0.010 from 153199): promotions at +{[p[0] for p in promos]} = the run's logged +426 / +801 / +1176")

ok(all(b2 >= b1 and b2 - b1 <= 1 for r in ALL_RUNGS for b1, b2 in zip(r, r[1:])), f"all {len(ALL_RUNGS)} scenarios: the rung never goes back and never skips (per-iteration trace)")
print("ALL PASS" if bad == 0 else f"FAIL ({bad})")
sys.exit(1 if bad else 0)
