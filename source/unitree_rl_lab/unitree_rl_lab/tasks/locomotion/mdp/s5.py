"""s5 terms (2026-11-20): study s5_stillness on lm7r_encup_t28w @178146 -- torso stillness + the arm hits to 0.

Operator 2026-11-20: "torso stability really will be our next study 100%" (the robot leaned back on hardware, the start of walking the
hardest -- he held it); arm hits: "I lean more towards 0". Every exam hit of the parent is torso_link x shoulder_roll_link = the upper arm
rolling INTO the torso, and Isaac cannot see that contact (torso_link collides as one 8 x 16 x 10 cm box at z 0.10-0.20, ~3 % of the real
torso's volume, >= 116 mm from shoulder_roll_link over the whole joint grid), so the arm terms work in joint space. Three terms, each the weight of one study lever; centre 0 = the term absent = the
parent's env bit for bit.

  torso_pitch_bonus      INCOME, an attractor (RL Bible Law 2 corollary: name the wanted state as a bounded positive kernel instead of
                         taxing the unwanted one), e = theta - theta0 (theta0 = 0 = upright in s5), r in [0, 1]:
                           shape "laplace" (s5's choice, operator 2026-11-20 ~18:25: "doesn't necessarily have to be a kernel. It can be
                           what works the best"): r = exp(-|e| / sigma) -- slope 1/sigma at the centre (it pulls right AT upright) and
                           (1/sigma) exp(-|e|/sigma) beyond, never flat where the robot leans 5-8 deg at a start (Rule 31's
                           linear-exponent kernel: widening by k turns an earned Y into Y^(1/k));
                           shape "gauss": r = exp(-(e / sigma)^2) -- flat at the centre AND beyond ~2.5 sigma (the flaw the operator named),
                         theta = torso_link pitch about the body y axis from the torso's projected gravity g_b = R^T (0, 0, -1):
                         theta = asin(g_x) = the ZYX-Euler pitch an IMU reports (R = Rz Ry Rx gives g_x = sin(pitch) whatever the
                         roll), POSITIVE = nose down (forward lean), negative = the back-lean the robot shows (the H1-2's IMU sits on
                         the torso: imu_link is merged into torso_link). UNGATED on purpose: the starts and stops
                         are where the robot leans back. sigma is sized from the parent's measured pitch (Rule 31, ~37 % earned).
  arm_adduct_hinge       HINGE PRICE (Law 2: a well-behaved robot pays ~0; Rule 33: judge it by where the angle SETTLES, not by its
                         income):  r = sum_{L,R} max(0, q_in - q_safe_side),  q_in = the shoulder roll angle measured INWARD (toward
                         the torso): left q_in = -q (limit [-0.38, 3.4], inward is negative), right q_in = +q (limit [-3.4, 0.38]).
                         q_safe per side = MuJoCo's contact onset of shoulder_roll_link with the torso mesh minus a margin (s5: onset
                         0.102-0.108 rad for shoulder pitch -0.1..1.2, q_safe 0.06). L1, not squared: squared, the onset pays 0.044^2 =
                         0.0019 per unit weight (w -30 -> -0.06 /s = a fee); L1 pays 0.044 |w| there (w -30 -> -1.3 /s).
  arm_side_sway_walking  VELOCITY PRICE on the SIDEWAYS arm axes only: r = sum over L/R shoulder_roll + shoulder_yaw of qd^2 while
                         |cmd| > cmd_thr (0.05, the observed command: Rule 32 -- never base velocity). The same op as
                         lm7j.arm_motion_walking on 4 of its 8 DOFs: arm_motion_walking (shoulder pitch / roll / yaw + elbow) and
                         joint_deviation_arms stay at the parent's weights, so a roll / yaw motion becomes strictly dearer than a
                         shoulder-pitch (front-back) swing. History (Rule 33's mirror): shoulder_roll_swing -0.2 was PAID at 28x its
                         weight per step (a fee, not a wall); lm6_arms (shoulder pitch freed + roll -1.0 on a soft scratch economy)
                         detonated. Here nothing is freed and no swing income is added (arm_antiphase earned 0 %).

Each term prints a proof line (JOB_VALIDITY 13; the measurement Rule 31 asks for): window statistics since its previous line -- every
10 iterations for the first 10 lines (a short smoke sees them), then every 100. They only read state; the reward is unaffected.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["torso_pitch", "torso_pitch_bonus", "shoulder_roll_inward", "arm_adduct_hinge", "arm_side_sway_walking", "S5_SIDE_JOINTS"]

PROOF_FAST, PROOF_FAST_N, PROOF_EVERY = 240, 10, 2400            # env steps (24 per PPO iteration)
SIGMA_PROBE_DEG = (1.0, 1.15, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)        # the earned fraction E[exp(-(e/s)^2)] the proof prints per candidate
LAPLACE_PROBE_DEG = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)                 # ... and E[exp(-|e|/s)]
PITCH_SHAPES = ("laplace", "gauss")
PITCH_BINS = (-20.0, 20.0, 0.25)                                  # deg: histogram for the percentiles (lo, hi, bin width)
QIN_BINS = (-1.5, 0.5, 0.01)                                      # rad
ROLL_JOINTS = ("left_shoulder_roll_joint", "right_shoulder_roll_joint")
S5_SIDE_JOINTS = (".*_shoulder_roll_joint", ".*_shoulder_yaw_joint")   # arm_side_sway_walking's 4 DOFs


# ── helpers ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _cmd_norm(env, command_name: str) -> torch.Tensor:
    return torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)


def torso_pitch(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """(N,) torso pitch in rad = the ZYX-Euler pitch (the IMU's rpy[1]) from the body's projected gravity: asin(g_x); + = nose down."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.body_quat_w[:, asset_cfg.body_ids, :].reshape(-1, 4)
    down = torch.zeros(q.shape[0], 3, device=q.device, dtype=q.dtype)
    down[:, 2] = -1.0
    g = quat_apply_inverse(q, down)
    return torch.asin(torch.clamp(g[:, 0], -1.0, 1.0))


def _proof_state(env, key: str) -> dict:
    """per-term proof accumulators on the env (a window between two proof lines)"""
    st = getattr(env, "_s5_proof", None)
    if st is None:
        st = env._s5_proof = {}
    return st.setdefault(key, {"n_lines": 0, "next": None, "acc": {}})


def _proof_due(st: dict, step: int) -> bool:
    if st["next"] is None:
        st["next"] = step + PROOF_FAST
    if step < st["next"]:
        return False
    st["n_lines"] += 1
    st["next"] = step + (PROOF_FAST if st["n_lines"] < PROOF_FAST_N else PROOF_EVERY)
    return True


def _acc(st: dict, name: str, value: torch.Tensor) -> None:
    a = st["acc"]
    a[name] = a[name] + value if name in a else value.clone()


def _hist(x: torch.Tensor, bins: tuple, w: torch.Tensor) -> torch.Tensor:
    """weighted histogram (w = the 0/1 mask as float64): no boolean indexing, so no GPU sync per step"""
    lo, hi, bw = bins
    nb = int(round((hi - lo) / bw))
    idx = ((x - lo) / bw).floor().clamp(0, nb - 1).long()
    return torch.bincount(idx, weights=w, minlength=nb)


def _pct(h: torch.Tensor, bins: tuple, qs=(0.05, 0.5, 0.95)) -> list:
    lo, _, bw = bins
    n = float(h.sum())
    if n <= 0:
        return [float("nan")] * len(qs)
    c = torch.cumsum(h, 0) / n
    return [lo + bw * (int(torch.searchsorted(c, torch.tensor(q, dtype=c.dtype, device=c.device))) + 0.5) for q in qs]


# ── torso_pitch_bonus ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def torso_pitch_bonus(env: "ManagerBasedRLEnv", sigma: float, theta0: float = 0.0, shape: str = "laplace",
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"), command_name: str = "base_velocity",
                      cmd_thr: float = 0.03) -> torch.Tensor:
    """e = theta - theta0 with theta = torso_link pitch (rad, see torso_pitch); shape "laplace": exp(-|e| / sigma), "gauss":
    exp(-(e / sigma)^2). UNGATED. POSITIVE weight (an income, ceiling 1/step). theta0 = the centre (rad; 0 = upright; s5 keeps 0 --
    Isaac's parent walks +1.4 deg nose down where the exam's MuJoCo shows -1.6, so a centre offset is the knob for that sim2sim gap).
    command_name / cmd_thr only split the proof line into walking (|cmd| >= cmd_thr) and standing; they do not gate the reward."""
    assert shape in PITCH_SHAPES, f"[s5] torso_pitch_bonus shape {shape!r} not in {PITCH_SHAPES}"
    th = torso_pitch(env, asset_cfg)
    e = (th - float(theta0)) / float(sigma)
    k = torch.exp(-torch.abs(e)) if shape == "laplace" else torch.exp(-torch.square(e))
    r = torch.nan_to_num(k, nan=0.0)
    _torso_pitch_proof(env, th, r, float(sigma), command_name, cmd_thr, float(theta0), shape)
    return r


def _torso_pitch_proof(env, th, r, sigma, command_name, cmd_thr, theta0=0.0, shape="laplace"):
    st = _proof_state(env, "torso_pitch")
    walk = _cmd_norm(env, command_name) >= cmd_thr
    deg = torch.rad2deg(torch.nan_to_num(th, nan=0.0)).to(torch.float64)
    r64 = r.to(torch.float64)
    ed = (deg - math.degrees(theta0)).unsqueeze(1)
    sig = torch.exp(-torch.square(ed / torch.tensor(SIGMA_PROBE_DEG, dtype=torch.float64, device=deg.device)))
    lap = torch.exp(-torch.abs(ed) / torch.tensor(LAPLACE_PROBE_DEG, dtype=torch.float64, device=deg.device))
    for tag, m in (("walk", walk), ("stand", ~walk)):
        w = m.to(torch.float64)
        _acc(st, f"{tag}_n", w.sum())
        _acc(st, f"{tag}_s1", (deg * w).sum())
        _acc(st, f"{tag}_s2", (deg * deg * w).sum())
        _acc(st, f"{tag}_r", (r64 * w).sum())
        _acc(st, f"{tag}_h", _hist(deg, PITCH_BINS, w))
        _acc(st, f"{tag}_sig", (sig * w.unsqueeze(1)).sum(dim=0))
        _acc(st, f"{tag}_lap", (lap * w.unsqueeze(1)).sum(dim=0))
        _acc(st, f"{tag}_abs", (ed.squeeze(1).abs() * w).sum())
    if not _proof_due(st, int(env.common_step_counter)):
        return
    a, parts = st["acc"], []
    for tag in ("walk", "stand"):
        n = float(a[f"{tag}_n"])
        if n <= 0:
            parts.append(f"{tag}: no samples")
            continue
        mu = float(a[f"{tag}_s1"]) / n
        sd = math.sqrt(max(float(a[f"{tag}_s2"]) / n - mu * mu, 0.0))
        p5, p50, p95 = _pct(a[f"{tag}_h"], PITCH_BINS)
        fr, fl = (a[f"{tag}_sig"] / n).tolist(), (a[f"{tag}_lap"] / n).tolist()
        parts.append(f"{tag} n {int(n)}: pitch mean {mu:+.2f} sd {sd:.2f} |e| {float(a[f'{tag}_abs']) / n:.2f} p5/50/95 {p5:+.1f}/{p50:+.1f}/{p95:+.1f} deg, "
                     f"earned {float(a[f'{tag}_r']) / n:.3f} ({shape}, sigma {math.degrees(sigma):.2f} deg); gauss "
                     + " ".join(f"{s:g}:{f:.2f}" for s, f in zip(SIGMA_PROBE_DEG, fr)) + "; laplace "
                     + " ".join(f"{s:g}:{f:.2f}" for s, f in zip(LAPLACE_PROBE_DEG, fl)))
    print(f"[s5] torso_pitch (+ = nose down) window to tick {int(env.common_step_counter)}: " + " | ".join(parts), flush=True)
    st["acc"] = {}


# ── arm_adduct_hinge ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _roll_ids_signs(env, asset_cfg: SceneEntityCfg, q_safe_left: float, q_safe_right: float):
    """(joint ids, inward sign, q_safe) per roll joint, resolved by NAME once (left: inward = -q, right: inward = +q)"""
    key = (asset_cfg.name, float(q_safe_left), float(q_safe_right))
    c = getattr(env, "_s5_adduct", None)
    if c is not None and c[0] == key:
        return c[1]
    asset = env.scene[asset_cfg.name]
    ids, names = asset.find_joints(list(ROLL_JOINTS), preserve_order=True)
    assert list(names) == list(ROLL_JOINTS), f"[s5] arm_adduct_hinge: the shoulder roll joints, got {names}"
    dev = asset.data.joint_pos.device
    sign = torch.tensor([-1.0, 1.0], device=dev)
    qs = torch.tensor([float(q_safe_left), float(q_safe_right)], device=dev)
    out = (list(ids), sign, qs)
    env._s5_adduct = (key, out)
    print(f"[s5] arm_adduct_hinge: joints {list(names)} ids {list(ids)}, inward sign {sign.tolist()}, q_safe (inward, rad) "
          f"left {float(q_safe_left):.4f} right {float(q_safe_right):.4f}", flush=True)
    return out


def shoulder_roll_inward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), q_safe_left: float = 0.0, q_safe_right: float = 0.0):
    """(N, 2) [left, right] shoulder roll angle measured toward the torso (rad); + = inward"""
    ids, sign, _ = _roll_ids_signs(env, asset_cfg, q_safe_left, q_safe_right)
    return env.scene[asset_cfg.name].data.joint_pos[:, ids] * sign


def arm_adduct_hinge(env: "ManagerBasedRLEnv", q_safe_left: float, q_safe_right: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     command_name: str = "base_velocity", cmd_thr: float = 0.03) -> torch.Tensor:
    """sum over the two shoulder rolls of max(0, q_in - q_safe_side) (rad), UNGATED. NEGATIVE weight. A linear hinge: 0 inside the safe
    range, slope |w| per rad past it (no flat region where it binds). The joints are found by name; asset_cfg only names the asset.
    command_name / cmd_thr only split the proof line."""
    ids, sign, qs = _roll_ids_signs(env, asset_cfg, q_safe_left, q_safe_right)
    q_in = env.scene[asset_cfg.name].data.joint_pos[:, ids] * sign
    over = torch.clamp(q_in - qs, min=0.0)
    r = torch.nan_to_num(over.sum(dim=1), nan=0.0)
    _arm_adduct_proof(env, q_in, over, r, command_name, cmd_thr)
    return r


def _arm_adduct_proof(env, q_in, over, r, command_name, cmd_thr):
    st = _proof_state(env, "arm_adduct")
    walk = _cmd_norm(env, command_name) >= cmd_thr
    q = torch.nan_to_num(q_in, nan=0.0).to(torch.float64)
    r64, past = r.to(torch.float64), (over > 0).to(torch.float64)
    for tag, m in (("walk", walk), ("stand", ~walk)):
        w = m.to(torch.float64)
        _acc(st, f"{tag}_n", w.sum())
        _acc(st, f"{tag}_r", (r64 * w).sum())
        for k, side in enumerate(("L", "R")):
            _acc(st, f"{tag}_{side}_s1", (q[:, k] * w).sum())
            _acc(st, f"{tag}_{side}_over", (past[:, k] * w).sum())
            _acc(st, f"{tag}_{side}_h", _hist(q[:, k], QIN_BINS, w))
            mx = torch.where(m, q[:, k], torch.full_like(q[:, k], -9.0)).max()
            a = st["acc"]
            a[f"{tag}_{side}_max"] = torch.maximum(a[f"{tag}_{side}_max"], mx) if f"{tag}_{side}_max" in a else mx
    if not _proof_due(st, int(env.common_step_counter)):
        return
    a, parts = st["acc"], []
    for tag in ("walk", "stand"):
        n = float(a[f"{tag}_n"])
        if n <= 0:
            parts.append(f"{tag}: no samples")
            continue
        sides = []
        for side in ("L", "R"):
            p50, p95, p99 = _pct(a[f"{tag}_{side}_h"], QIN_BINS, (0.5, 0.95, 0.99))
            sides.append(f"{side} q_in mean {float(a[f'{tag}_{side}_s1']) / n:+.3f} p50/95/99 {p50:+.3f}/{p95:+.3f}/{p99:+.3f} max "
                         f"{float(a[f'{tag}_{side}_max']):+.3f} rad, past q_safe {100.0 * float(a[f'{tag}_{side}_over']) / n:.2f} %")
        parts.append(f"{tag} n {int(n)}: hinge mean {float(a[f'{tag}_r']) / n:.5f} rad; " + "; ".join(sides))
    print(f"[s5] arm_adduct (q_in + = toward the torso) window to tick {int(env.common_step_counter)}: " + " | ".join(parts), flush=True)
    st["acc"] = {}


# ── arm_side_sway_walking ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def arm_side_sway_walking(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, command_name: str = "base_velocity",
                          cmd_thr: float = 0.05) -> torch.Tensor:
    """sum over asset_cfg.joint_ids (both shoulder_roll + shoulder_yaw = 4 DOFs) of joint_vel^2 while |cmd| > cmd_thr (the observed
    command, Bible rule 32; the strict '>' and 0.05 of arm_motion_walking). NEGATIVE weight. An L2 price, no kernel."""
    asset = env.scene[asset_cfg.name]
    v = asset.data.joint_vel[:, asset_cfg.joint_ids]
    walking = (_cmd_norm(env, command_name) > cmd_thr).float()
    r = torch.nan_to_num(torch.sum(v * v, dim=1) * walking, nan=0.0)
    _arm_side_sway_proof(env, asset, walking > 0, r)
    return r


def _arm_side_sway_proof(env, asset, walk, r):
    st = _proof_state(env, "arm_side_sway")
    grp = getattr(env, "_s5_sway_groups", None)
    if grp is None:
        grp = {}
        for g, pat in (("roll", ".*_shoulder_roll_joint"), ("yaw", ".*_shoulder_yaw_joint"), ("pitch", ".*_shoulder_pitch_joint")):
            grp[g] = asset.find_joints(pat)[0]
        env._s5_sway_groups = grp
    v2 = torch.square(torch.nan_to_num(asset.data.joint_vel, nan=0.0)).to(torch.float64)
    w = walk.to(torch.float64)
    _acc(st, "n", torch.ones_like(w).sum())
    _acc(st, "n_walk", w.sum())
    _acc(st, "r", r.to(torch.float64).sum())
    for g, ids in grp.items():
        per_env = v2[:, ids].sum(dim=1)
        _acc(st, f"walk_{g}", (per_env * w).sum())
        _acc(st, f"stand_{g}", (per_env * (1.0 - w)).sum())
    if not _proof_due(st, int(env.common_step_counter)):
        return
    a = st["acc"]
    n, nw = float(a["n"]), float(a["n_walk"])
    ns = n - nw
    per = lambda k, d: float(a[k]) / d if d > 0 else float("nan")
    print(f"[s5] arm_side_sway window to tick {int(env.common_step_counter)}: term mean {per('r', n):.4f} /step over all "
          f"({per('r', nw):.4f} per walking step, walking {100.0 * nw / max(n, 1.0):.1f} % of {int(n)}); sum qd^2 (rad^2/s^2, both arms) "
          f"walking roll {per('walk_roll', nw):.4f} yaw {per('walk_yaw', nw):.4f} pitch {per('walk_pitch', nw):.4f} | standing roll "
          f"{per('stand_roll', ns):.4f} yaw {per('stand_yaw', ns):.4f} pitch {per('stand_pitch', ns):.4f}", flush=True)
    st["acc"] = {}
