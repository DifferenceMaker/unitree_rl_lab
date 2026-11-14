"""s3 lever unit tests (no simulator): (a) the single-axis sampler bucket, (b) the command-relative tracking kernel.

The REAL repo modules (velocity_command.py, lm7i.py) are loaded with stubbed isaaclab imports (isaaclab needs pxr = the app);
the pre-change code is loaded the same way from `git show <OLD_REV>:...`; IsaacLab's own UniformVelocityCommand._resample_command
and track_ang_vel_z_exp are extracted from the IsaacLab source with ast and exec'd, so the base draws are the real ones.
Usage (isaacsim conda env): python -I scripts/test_s3_levers.py [--device cuda]   -> "ALL PASS"
"""
import argparse
import ast
import dataclasses
import importlib.util
import math
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path.home() / "Projects/robot_projects/repos/unitree_rl_lab-ik"
PKG = REPO / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp"
ISAAC = Path.home() / "Projects/robot_projects/repos/IsaacLab/source/isaaclab/isaaclab"
OLD_REV = "69cce84"


def extract(src: str, qualname: str) -> str:
    """source of a top-level function or Class.method from a module source."""
    tree = ast.parse(src)
    parts = qualname.split(".")
    nodes = tree.body
    for i, p in enumerate(parts):
        node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == p)
        nodes = node.body
    seg = ast.get_source_segment(src, node)
    if len(parts) > 1:   # dedent a method
        import textwrap
        seg = textwrap.dedent(seg)
    return seg


# ── stubs ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def base_resample():
    src = (ISAAC / "envs/mdp/commands/velocity_command.py").read_text()
    ns = {"torch": torch, "Sequence": list}
    exec(extract(src, "UniformVelocityCommand._resample_command"), ns)
    return ns["_resample_command"]


class UniformVelocityCommandStub:
    _resample_command = base_resample()


class UniformVelocityCommandCfgStub:
    class Ranges:
        pass


def yaw_quat(q):   # wxyz
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = torch.zeros_like(q)
    out[:, 0], out[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return out


def quat_apply_inverse(q, v):
    w, xyz = q[:, :1], -q[:, 1:]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def install_stubs():
    m = {n: types.ModuleType(n) for n in ("isaaclab", "isaaclab.envs", "isaaclab.envs.mdp", "isaaclab.utils", "isaaclab.utils.math",
                                          "isaaclab.managers")}
    m["isaaclab.envs.mdp"].UniformVelocityCommand = UniformVelocityCommandStub
    m["isaaclab.envs.mdp"].UniformVelocityCommandCfg = UniformVelocityCommandCfgStub
    m["isaaclab.utils"].configclass = lambda c: c
    m["isaaclab.utils.math"].yaw_quat = yaw_quat
    m["isaaclab.utils.math"].quat_apply_inverse = quat_apply_inverse
    m["isaaclab.managers"].SceneEntityCfg = lambda name, **kw: SimpleNamespace(name=name, **kw)
    sys.modules.update(m)


def load(name: str, path: Path | None = None, src: str | None = None):
    if src is None:
        src = path.read_text()
    mod = types.ModuleType(name)
    mod.__file__ = str(path or name)
    sys.modules[name] = mod
    exec(compile(src, str(path or name), "exec"), mod.__dict__)
    return mod


def git_old(rel: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "show", f"{OLD_REV}:source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/{rel}"], text=True)


# ── fake command term ────────────────────────────────────────────────────────────────────────────────────────────────────────
def make_term(cls, n, dev, ranges, **cfgkw):
    t = object.__new__(cls)
    R = SimpleNamespace(lin_vel_x=ranges[0], lin_vel_y=ranges[1], ang_vel_z=ranges[2], heading=(-math.pi, math.pi))
    t.cfg = SimpleNamespace(ranges=R, heading_command=True, rel_heading_envs=0.5, rel_standing_envs=0.25, **cfgkw)
    t.device, t.num_envs = dev, n
    t.vel_command_b = torch.zeros(n, 3, device=dev)
    t.heading_target = torch.zeros(n, device=dev)
    t.is_heading_env = torch.zeros(n, dtype=torch.bool, device=dev)
    t.is_standing_env = torch.zeros(n, dtype=torch.bool, device=dev)
    return t


def state(t):
    return [t.vel_command_b.clone(), t.heading_target.clone(), t.is_heading_env.clone(), t.is_standing_env.clone()]


def rng_state(dev):
    return torch.cuda.get_rng_state() if dev.startswith("cuda") else torch.get_rng_state()


def _chk(c, line):
    if not c:
        print(f"   !! CHECK FAILED at test line {line}")
    return bool(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = ap.parse_args().device
    install_stubs()
    new_vc = load("vc_new", PKG / "commands/velocity_command.py")
    old_vc = load("vc_old", src=git_old("commands/velocity_command.py"))
    T1_RANGES = ((-0.8, 1.2), (-0.3, 0.3), (-0.8, 0.8))
    S3_RANGES = ((-0.5, 0.8), (-0.3, 0.3), (-0.6, 0.6))      # operator 2026-11-14: wz +-0.6
    ok = True

    # ── (a1) single_prob 0 -> bit-identical to the pre-change class, same seed, incl. the RNG state after ─────────────────────────
    print(f"== (a1) single_prob 0 vs pre-change CornerVelocityCommand ({OLD_REV}), device {dev}")
    configs = {
        "t1 (corner .25)": dict(corner_prob=0.25),
        "all off": dict(),
        "corner+slow+spin": dict(corner_prob=0.2, slow_prob=0.15, slow_band=0.15, spin_prob=0.1, spin_min=0.2),
        "slow only": dict(slow_prob=0.3, slow_band=0.3),
    }
    for name, kw in configs.items():
        for ranges in (T1_RANGES, S3_RANGES, ((-0.1, 0.1), (-0.1, 0.1), (-0.2, 0.2))):
            n = 4096
            a = make_term(old_vc.CornerVelocityCommand, n, dev, ranges, **kw)
            b = make_term(new_vc.CornerVelocityCommand, n, dev, ranges, single_prob=0.0, single_min=(0.03, 0.03, 0.05), **kw)
            same = True
            for seed in range(5):
                gen_ids = torch.Generator().manual_seed(100 + seed)
                ids_list = [torch.arange(n), torch.randperm(n, generator=gen_ids)[:777].sort().values, torch.tensor([3, 9, 10]), torch.arange(0)]
                for ids in ids_list:
                    ids = ids.to(dev)
                    torch.manual_seed(seed)
                    a._resample_command(ids)
                    ra = rng_state(dev).clone()
                    torch.manual_seed(seed)
                    b._resample_command(ids)
                    rb = rng_state(dev).clone()
                    same &= all(torch.equal(x, y) for x, y in zip(state(a), state(b))) and torch.equal(ra, rb)
            print(f"   {name:18s} ranges {ranges}: {'IDENTICAL (outputs + RNG state)' if same else 'DIFFERENT'}")
            ok &= _chk(same, 158)

    # ── (a2) 1e6 draws at single_prob 0.35, corner_prob 0.15 via the pure function ───────────────────────────────────────────────
    N = 1_000_000
    print(f"== (a2) {N} draws, single_prob 0.35, corner_prob 0.15, ranges vx {S3_RANGES[0]} vy {S3_RANGES[1]} wz {S3_RANGES[2]}")
    torch.manual_seed(7)
    cmd = torch.stack([torch.empty(N, device=dev).uniform_(*r) for r in S3_RANGES], dim=1)
    bucket = new_vc.sample_command_buckets(cmd, *S3_RANGES, corner_prob=0.15, single_prob=0.35, single_min=(0.03, 0.03, 0.05))
    cnt = torch.bincount(bucket, minlength=7).cpu()
    sh = cnt.double() / N
    print("   bucket shares: " + ", ".join(f"{nm} {100 * float(x):.3f} %" for nm, x in zip(new_vc.BUCKET_NAMES, sh)))
    single = bucket >= new_vc.BUCKET_SINGLE
    ns = int(single.sum())
    print(f"   single total {100 * ns / N:.3f} % (cfg 35), corner {100 * float(sh[1]):.3f} % (cfg 15), uniform {100 * float(sh[0]):.3f} % (cfg 50)")
    ax_share = [int((bucket == new_vc.BUCKET_SINGLE + a).sum()) / ns for a in range(3)]
    print(f"   per-axis share of single draws: vx {ax_share[0]:.4f} vy {ax_share[1]:.4f} wz {ax_share[2]:.4f} (1/3 = 0.3333)")
    for a, nm in enumerate(("vx", "vy", "wz")):
        rows = cmd[bucket == new_vc.BUCKET_SINGLE + a]
        others = [o for o in range(3) if o != a]
        nz_other = int((rows[:, others] != 0).sum())
        mag = rows[:, a].abs()
        pos = float((rows[:, a] > 0).double().mean())
        print(f"   single {nm}: n {len(rows)}, other two axes nonzero entries {nz_other} (must be 0), |c| min {float(mag.min()):.5f} "
              f"max {float(mag.max()):.5f}, + sign share {pos:.4f}")
        ok &= _chk(nz_other == 0, 182)
    vy = cmd[bucket == 5][:, 1].abs()
    p_vy = float(((vy >= 0.03) & (vy <= 0.10)).double().mean())
    exp_vy = math.log(0.10 / 0.03) / math.log(0.3 / 0.03)
    wz = cmd[bucket == 6][:, 2].abs()
    p_wz = float(((wz >= 0.10) & (wz <= 0.30)).double().mean())
    exp_wz = math.log(3) / math.log(S3_RANGES[2][1] / 0.05)
    print(f"   P(0.03<=|vy|<=0.10 | single vy) = {p_vy:.4f}  (expected ln(0.10/0.03)/ln(0.3/0.03) = {exp_vy:.4f})")
    print(f"   P(0.10<=|wz|<=0.30 | single wz) = {p_wz:.4f}  (expected ln 3/ln({S3_RANGES[2][1]}/0.05) = {exp_wz:.4f})")
    vx = cmd[bucket == 4][:, 0]
    fwd, bwd = vx[vx > 0], -vx[vx < 0]
    p_f = float((fwd <= 0.10).double().mean()); e_f = math.log(0.10 / 0.03) / math.log(0.8 / 0.03)
    p_b = float((bwd <= 0.10).double().mean()); e_b = math.log(0.10 / 0.03) / math.log(0.5 / 0.03)
    print(f"   single vx forward: n {len(fwd)}, max {float(fwd.max()):.5f} (edge 0.8), min {float(fwd.min()):.5f}; P(|vx|<=0.10) {p_f:.4f} (exp {e_f:.4f})")
    print(f"   single vx backward: n {len(bwd)}, max {float(bwd.max()):.5f} (edge 0.5), min {float(bwd.min()):.5f}; P(|vx|<=0.10) {p_b:.4f} (exp {e_b:.4f})")
    ok &= _chk(abs(p_vy - exp_vy) < 0.01 and abs(p_wz - exp_wz) < 0.01 and float(fwd.max()) <= 0.8 and float(bwd.max()) <= 0.5, 197)
    ok &= _chk(all(abs(s - 1 / 3) < 0.005 for s in ax_share) and abs(ns / N - 0.35) < 0.003 and abs(float(sh[1]) - 0.15) < 0.003, 198)

    # ── (a3) the CLASS uses the same function: same seed -> the same commands; heading flag off for every single draw ──────────
    print("== (a3) CornerVelocityCommand at single_prob 0.35 / corner 0.15 (the class path)")
    t = make_term(new_vc.CornerVelocityCommand, N, dev, S3_RANGES, corner_prob=0.15, single_prob=0.35, single_min=(0.03, 0.03, 0.05))
    torch.manual_seed(7)
    t._resample_command(torch.arange(N, device=dev))
    # replay: the base draws, then the pure function, on the same seed
    torch.manual_seed(7)
    r = torch.empty(N, device=dev)
    c2 = torch.zeros(N, 3, device=dev)
    for a, rng in enumerate(S3_RANGES):
        c2[:, a] = r.uniform_(*rng)
    r.uniform_(-math.pi, math.pi); hd = r.uniform_(0.0, 1.0) <= 0.5; r.uniform_(0.0, 1.0)
    b2 = new_vc.sample_command_buckets(c2, *S3_RANGES, corner_prob=0.15, single_prob=0.35, single_min=(0.03, 0.03, 0.05))
    same_cmd = torch.equal(c2, t.vel_command_b)
    sgl = b2 >= new_vc.BUCKET_SINGLE
    head_single = int(t.is_heading_env[sgl].sum())
    head_other = float(t.is_heading_env[~sgl].double().mean())
    zeros2 = int(((t.vel_command_b == 0).sum(dim=1) == 2).sum())
    print(f"   class commands == base draws + sample_command_buckets on the same seed: {same_cmd}")
    print(f"   heading flag ON among single draws: {head_single} of {int(sgl.sum())} (must be 0); among the rest: {100 * head_other:.2f} % (rel_heading_envs 50)")
    print(f"   rows with exactly two zero axes: {zeros2} = {100 * zeros2 / N:.3f} % (the single bucket)")
    ok &= _chk(same_cmd and head_single == 0 and zeros2 == int(sgl.sum()), 221)
    t._bucket_printed, t._bucket_counts = False, None   # the proof print (fires once >= 2 x num_envs; forced here: num_envs 1)
    t.num_envs = 1
    t._bucket_proof(b2, dict(corner_prob=0.15, slow_prob=0.0, spin_prob=0.0, single_prob=0.35, single_min=(0.03, 0.03, 0.05)))

    # ── (b) kernel ────────────────────────────────────────────────────────────────────────────────────────────────────────────
    print("== (b) command-relative kernel")
    new_k = load("lm7i_new", PKG / "lm7i.py")
    old_k = load("lm7i_old", src=git_old("lm7i.py"))
    isaac_src = (ISAAC / "envs/mdp/rewards.py").read_text()
    ns = {"torch": torch, "SceneEntityCfg": sys.modules["isaaclab.managers"].SceneEntityCfg}
    exec("from __future__ import annotations\n" + extract(isaac_src, "track_ang_vel_z_exp"), ns)
    isaac_ang = ns["track_ang_vel_z_exp"]

    def env_of(cmd, v_w, wz_b, quat):
        data = SimpleNamespace(root_quat_w=quat, root_lin_vel_w=v_w, root_ang_vel_b=torch.stack([torch.zeros_like(wz_b)] * 2 + [wz_b], 1))
        return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, command_manager=SimpleNamespace(get_command=lambda name: cmd))

    # unit numbers: sigma 0.0817, cref 0.16, standing still under a pure vy 0.05 command
    z = torch.zeros(1, 3, device=dev)
    q = torch.tensor([[1.0, 0, 0, 0]], device=dev)
    e = env_of(torch.tensor([[0.0, 0.05, 0.0]], device=dev), z, torch.zeros(1, device=dev), q)
    r_rel = float(new_k.track_lin_vel_xy_split(e, std=0.0817, cref=0.16))
    r_off = float(new_k.track_lin_vel_xy_split(e, std=0.0817))
    s_eff = float(new_k.cmd_relative_std(0.0817, torch.tensor(0.05), 0.16, 0.25))
    print(f"   lin: sigma_eff {s_eff:.5f} (0.0817 x 0.3125 = {0.0817 * 0.3125:.5f}); reward cref 0.16 {r_rel:.4f} (exp(-(0.05/0.02553)^2) = "
          f"{math.exp(-(0.05 / (0.0817 * 0.3125)) ** 2):.4f}); cref 0 {r_off:.4f} (exp(-(0.05/0.0817)^2) = {math.exp(-(0.05 / 0.0817) ** 2):.4f})")
    ok &= _chk(abs(r_rel - math.exp(-(0.05 / (0.0817 * 0.3125)) ** 2)) < 1e-5 and abs(r_off - math.exp(-(0.05 / 0.0817) ** 2)) < 1e-5, 248)
    e = env_of(torch.tensor([[0.0, 0.0, 0.10]], device=dev), z, torch.zeros(1, device=dev), q)
    a_rel = float(new_k.track_ang_vel_z_exp_rel(e, std=0.0817, command_name="base_velocity", cref=0.25))
    a_off = float(new_k.track_ang_vel_z_exp_rel(e, std=0.0817, command_name="base_velocity"))
    print(f"   ang: pure wz 0.10, standing still, cref 0.25: sigma_eff {0.0817 * 0.4:.5f}, reward {a_rel:.5f} (exp {math.exp(-(0.1 / (0.0817 * 0.4)) ** 2):.5f}); "
          f"cref 0 {a_off:.4f} (exp {math.exp(-(0.1 / 0.0817) ** 2):.4f})")
    e = env_of(torch.tensor([[0.0, 0.0, 0.0]], device=dev), z, torch.tensor([0.05], device=dev), q)
    a_st_rel = float(new_k.track_ang_vel_z_exp_rel(e, std=0.0817, command_name="base_velocity", cref=0.25))
    a_st_old = float(isaac_ang(e, std=0.0817, command_name="base_velocity"))
    print(f"   ang standing env (cmd 0), yaw rate 0.05: cref 0.25 {a_st_rel:.5f} == isaaclab {a_st_old:.5f} (standing branch unchanged)")
    e = env_of(torch.tensor([[0.5, 0.0, 0.0]], device=dev), z, torch.tensor([0.05], device=dev), q)
    a_w = float(new_k.track_ang_vel_z_exp_rel(e, std=0.0817, command_name="base_velocity", cref=0.25))
    print(f"   ang straight walk vx 0.5 (wz cmd 0), yaw wobble 0.05: cref 0.25 {a_w:.5f} (sigma x fmin 0.0204) vs cref 0 {math.exp(-(0.05 / 0.0817) ** 2):.4f}")
    ok &= _chk(abs(a_st_rel - a_st_old) == 0.0, 261)

    # cref 0 == old functions on random inputs, max abs diff
    g = torch.Generator(device=dev).manual_seed(3)
    n = 200_000
    cmd = torch.rand(n, 3, device=dev, generator=g) * 2 - 1
    cmd[: n // 4] = 0.0                                   # standing rows
    cmd[n // 4: n // 2] *= 0.05                           # small commands around the gate
    v_w = (torch.rand(n, 3, device=dev, generator=g) * 2 - 1) * 1.5
    wz_b = (torch.rand(n, device=dev, generator=g) * 2 - 1) * 1.5
    quat = torch.nn.functional.normalize(torch.randn(n, 4, device=dev, generator=g), dim=1)
    e = env_of(cmd, v_w, wz_b, quat)
    for std in (0.15, 0.1, 0.0817):
        d_lin = float((new_k.track_lin_vel_xy_split(e, std=std, std_stand=0.5, cmd_thr=0.03)
                       - old_k.track_lin_vel_xy_split(e, std=std, std_stand=0.5, cmd_thr=0.03)).abs().max())
        d_lin_explicit = float((new_k.track_lin_vel_xy_split(e, std=std, std_stand=0.5, cmd_thr=0.03, cref=0.0, fmin=0.25)
                                - old_k.track_lin_vel_xy_split(e, std=std, std_stand=0.5, cmd_thr=0.03)).abs().max())
        d_ang = float((new_k.track_ang_vel_z_exp_rel(e, std=std, command_name="base_velocity", cref=0.0, fmin=0.25)
                       - isaac_ang(e, std=std, command_name="base_velocity")).abs().max())
        eq = torch.equal(new_k.track_lin_vel_xy_split(e, std=std), old_k.track_lin_vel_xy_split(e, std=std)) and \
            torch.equal(new_k.track_ang_vel_z_exp_rel(e, std=std, command_name="base_velocity"), isaac_ang(e, std=std, command_name="base_velocity"))
        print(f"   cref 0 vs old, {n} random rows, std {std}: lin max|diff| {d_lin} (params absent) / {d_lin_explicit} (cref 0, fmin .25 given); "
              f"ang vs isaaclab track_ang_vel_z_exp {d_ang}; torch.equal {eq}")
        ok &= _chk(d_lin == 0.0 and d_lin_explicit == 0.0 and d_ang == 0.0 and eq, 284)
    # cref > 0: rows the narrowing does not touch (standing, or the term's command component >= cref) equal the old kernel exactly
    std = 0.0817
    nl = new_k.track_lin_vel_xy_split(e, std=std, cref=0.16, fmin=0.25)
    ol = old_k.track_lin_vel_xy_split(e, std=std)
    mov = torch.linalg.norm(cmd, dim=1) >= 0.03
    touched_l = mov & (torch.linalg.norm(cmd[:, :2], dim=1) < 0.16)
    na = new_k.track_ang_vel_z_exp_rel(e, std=std, command_name="base_velocity", cref=0.25, fmin=0.25)
    oa = isaac_ang(e, std=std, command_name="base_velocity")
    touched_a = mov & (cmd[:, 2].abs() < 0.25)
    dl, da = float((nl - ol)[~touched_l].abs().max()), float((na - oa)[~touched_a].abs().max())
    print(f"   cref>0 (lin .16, ang .25): untouched rows max|diff| vs old: lin {dl} over {int((~touched_l).sum())} rows, ang {da} over "
          f"{int((~touched_a).sum())} rows; narrowed rows: lin {int(touched_l.sum())} (all <= old: {bool((nl[touched_l] <= ol[touched_l]).all())}), "
          f"ang {int(touched_a.sum())} (all <= old: {bool((na[touched_a] <= oa[touched_a]).all())})")
    ok &= _chk(dl == 0.0 and da == 0.0, 0)
    # composition with the ladder: the width follows the std passed at call time
    widths = [float(new_k.cmd_relative_std(s, torch.tensor(0.05), 0.16, 0.25)) for s in (0.15, 0.1, 0.0817)]
    print(f"   sigma_eff at |c| 0.05, cref 0.16 along the ladder (0.15, 0.10, 0.0817): {[round(w, 5) for w in widths]} (= std x 0.3125)")
    print("ALL PASS" if ok else "SOME CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
