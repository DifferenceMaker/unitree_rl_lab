"""s5 term unit tests (no simulator): torso_pitch / torso_pitch_bonus, arm_adduct_hinge, arm_side_sway_walking on synthetic tensors.

The REAL mdp/s5.py is loaded with stubbed isaaclab imports; isaaclab's own quat_apply_inverse and lm7j's arm_motion_walking + _walking
are extracted from their sources with ast and exec'd, so the frame math and the reference op are the real ones.
Usage (isaacsim conda env): python -I scripts/test_s5_terms.py [--device cuda]   -> "ALL PASS"
"""
import argparse
import ast
import io
import math
import re
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path.home() / "Projects/robot_projects/repos/unitree_rl_lab-ik"
PKG = REPO / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp"
ISAAC = Path.home() / "Projects/robot_projects/repos/IsaacLab/source/isaaclab/isaaclab"
JOINTS = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
          "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", "torso_joint"]
FAILS = []


def extract(src: str, name: str) -> str:
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, node)          # the def without its decorator (torch.jit.script)


def check(c, msg):
    print(("  ok  " if c else "  FAIL ") + msg)
    if not c:
        FAILS.append(msg)


def install_stubs():
    ns = {"torch": torch}
    exec(extract((ISAAC / "utils/math.py").read_text(), "quat_apply_inverse"), ns)
    m = {n: types.ModuleType(n) for n in ("isaaclab", "isaaclab.utils", "isaaclab.utils.math", "isaaclab.managers", "isaaclab.envs")}
    m["isaaclab.utils.math"].quat_apply_inverse = ns["quat_apply_inverse"]
    m["isaaclab.managers"].SceneEntityCfg = lambda name, **kw: SimpleNamespace(name=name, **kw)
    sys.modules.update(m)
    return ns["quat_apply_inverse"]


def load_s5():
    mod = types.ModuleType("s5")
    mod.__file__ = str(PKG / "s5.py")
    exec(compile((PKG / "s5.py").read_text(), mod.__file__, "exec"), mod.__dict__)
    return mod


def lm7j_ref():
    src = (PKG / "lm7j.py").read_text()
    ns = {"torch": torch, "SceneEntityCfg": None}
    exec(extract(src, "_walking"), ns)
    exec(extract(src, "arm_motion_walking"), ns)
    return ns["arm_motion_walking"]


class Robot:
    def __init__(self, n, dev):
        self.joint_names = list(JOINTS)
        self.data = SimpleNamespace(body_quat_w=torch.zeros(n, 2, 4, device=dev), joint_pos=torch.zeros(n, len(JOINTS), device=dev),
                                    joint_vel=torch.zeros(n, len(JOINTS), device=dev))
        self.data.body_quat_w[..., 0] = 1.0

    def find_joints(self, keys, preserve_order=False):
        keys = [keys] if isinstance(keys, str) else list(keys)
        if preserve_order:
            hits = [(i, n) for k in keys for i, n in enumerate(self.joint_names) if re.fullmatch(k, n)]
        else:
            hits = [(i, n) for i, n in enumerate(self.joint_names) if any(re.fullmatch(k, n) for k in keys)]
        return [h[0] for h in hits], [h[1] for h in hits]


def make_env(n, dev):
    robot = Robot(n, dev)
    cmd = torch.zeros(n, 3, device=dev)
    env = SimpleNamespace(scene={"robot": robot}, common_step_counter=0, num_envs=n, device=dev,
                          command_manager=SimpleNamespace(get_command=lambda name: cmd))
    return env, robot, cmd


def quat_axis(axis, ang):
    """(w, x, y, z) of a rotation by ang about a unit axis"""
    a = torch.tensor(axis, dtype=torch.float32)
    a = a / a.norm()
    return torch.cat([torch.tensor([math.cos(ang / 2)]), a * math.sin(ang / 2)])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return torch.tensor([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    dev = ap.parse_args().device
    install_stubs()
    s5 = load_s5()
    ref_arm_motion = lm7j_ref()
    torso = SimpleNamespace(name="robot", body_ids=[1])

    print("torso_pitch: the sign and the frame")
    env, robot, cmd = make_env(6, dev)
    d = math.radians
    yaw90 = quat_axis((0, 0, 1), d(90))
    quats = [quat_axis((0, 1, 0), d(5)),                          # 5 deg nose down
             quat_axis((0, 1, 0), d(-8)),                         # 8 deg back-lean (the robot's start lean)
             quat_axis((1, 0, 0), d(10)),                         # pure roll: pitch 0
             qmul(yaw90, quat_axis((0, 1, 0), d(-3))),            # heading 90 deg, then 3 deg back (body frame): yaw must not leak
             torch.tensor([1.0, 0, 0, 0]),                        # upright
             qmul(quat_axis((0, 1, 0), d(4)), quat_axis((1, 0, 0), d(6)))]   # pitch 4 then roll 6 (ZYX intrinsic): Euler pitch 4 (atan2 would say 4.02)
    for i, q in enumerate(quats):
        robot.data.body_quat_w[i, 1] = q.to(dev)
    th = torch.rad2deg(s5.torso_pitch(env, torso)).cpu()
    want = [5.0, -8.0, 0.0, -3.0, 0.0, 4.0]
    check(torch.allclose(th, torch.tensor(want), atol=1e-4), f"pitch {th.tolist()} == {want} (+ = nose down, yaw / roll do not leak)")

    print("torso_pitch_bonus: exp(-(theta/sigma)^2), ungated, bounded")
    sig = d(3.0)
    with redirect_stdout(io.StringIO()):
        r = s5.torso_pitch_bonus(env, sigma=sig, shape="gauss", asset_cfg=torso).cpu()
    k = torch.exp(-torch.square(torch.tensor(want) / 3.0))
    check(torch.allclose(r, k, atol=1e-5), f"kernel {r.tolist()} == exp(-(theta/3deg)^2)")
    check(abs(float(r[4]) - 1.0) < 1e-6 and float(r.min()) >= 0.0 and float(r.max()) <= 1.0, "ceiling 1.0 upright, in [0, 1]")
    with redirect_stdout(io.StringIO()):
        rl = s5.torso_pitch_bonus(env, sigma=sig, shape="laplace", asset_cfg=torso).cpu()
    kl = torch.exp(-torch.tensor(want).abs() / 3.0)
    check(torch.allclose(rl, kl, atol=1e-5), f"laplace {[round(x, 4) for x in rl.tolist()]} == exp(-|theta|/3deg)")
    check(float(rl[1]) > 0.06 and float(k[1]) < 0.001, f"at the -8 deg start lean the laplace kernel still pays {float(rl[1]):.3f} (gauss {float(k[1]):.5f})")
    try:
        with redirect_stdout(io.StringIO()):
            s5.torso_pitch_bonus(env, sigma=sig, shape="tent", asset_cfg=torso)
        check(False, "an unknown shape is refused")
    except AssertionError:
        check(True, "an unknown shape is refused")
    with redirect_stdout(io.StringIO()):
        r0 = s5.torso_pitch_bonus(env, sigma=sig, theta0=d(-8.0), shape="gauss", asset_cfg=torso).cpu()
    check(abs(float(r0[1]) - 1.0) < 1e-5 and abs(float(r0[4]) - math.exp(-(8 / 3) ** 2)) < 1e-5, "theta0 moves the kernel's centre (theta0 -8 deg: row 1 pays 1.0)")
    cmd[:] = 0.0
    cmd[0, 0] = 0.5
    with redirect_stdout(io.StringIO()):
        r2 = s5.torso_pitch_bonus(env, sigma=sig, shape="gauss", asset_cfg=torso).cpu()
    check(torch.equal(r, r2), "the command does not change the reward (ungated: starts and stops are where the robot leans)")
    robot.data.body_quat_w[0, 1] = torch.tensor([float("nan")] * 4, device=dev)
    with redirect_stdout(io.StringIO()):
        r3 = s5.torso_pitch_bonus(env, sigma=sig, shape="gauss", asset_cfg=torso).cpu()
    check(bool(torch.isfinite(r3).all()) and float(r3[0]) == 0.0, "a NaN quaternion pays 0, not NaN (Bible rule 14)")

    print("arm_adduct_hinge: sum_{L,R} max(0, q_in - q_safe), q_in by side")
    env, robot, cmd = make_env(5, dev)
    jp = robot.data.joint_pos
    iL, iR = JOINTS.index("left_shoulder_roll_joint"), JOINTS.index("right_shoulder_roll_joint")
    rows = [(-0.30, 0.25), (0.40, -0.50), (-0.20, 0.20), (-0.21, 0.205), (0.0, 0.0)]   # (left q, right q)
    for i, (l, rr) in enumerate(rows):
        jp[i, iL], jp[i, iR] = l, rr
    cfg = SimpleNamespace(name="robot")
    with redirect_stdout(io.StringIO()) as out:
        h = s5.arm_adduct_hinge(env, q_safe_left=0.20, q_safe_right=0.20, asset_cfg=cfg).cpu()
    want = [0.10 + 0.05, 0.0, 0.0, 0.01 + 0.005, 0.0]
    check(torch.allclose(h, torch.tensor(want), atol=1e-6), f"hinge {[round(x, 4) for x in h.tolist()]} == {want} "
          "(left inward = -q, right inward = +q; outward and the safe band pay 0)")
    q_in = s5.shoulder_roll_inward(env, cfg, 0.20, 0.20).cpu()
    check(torch.allclose(q_in[0], torch.tensor([0.30, 0.25])) and torch.allclose(q_in[1], torch.tensor([-0.40, -0.50])), f"q_in rows 0/1 {q_in[:2].tolist()}")
    check("ids [1, 5]" in out.getvalue() and "inward sign [-1.0, 1.0]" in out.getvalue(), "joints resolved by name: " + out.getvalue().strip()[:120])
    with redirect_stdout(io.StringIO()):
        h2 = s5.arm_adduct_hinge(env, q_safe_left=0.30, q_safe_right=0.10, asset_cfg=cfg).cpu()
    check(abs(float(h2[0]) - (0.0 + 0.15)) < 1e-6, f"per-side q_safe (L 0.30, R 0.10) row 0 = {float(h2[0]):.4f} == 0.15")

    print("arm_side_sway_walking: == lm7j.arm_motion_walking on the 4 sideways DOFs; strict '>' gate on the command")
    env, robot, cmd = make_env(4, dev)
    g = torch.Generator().manual_seed(0)
    robot.data.joint_vel[:] = torch.randn(4, len(JOINTS), generator=g).to(dev)
    cmd[:] = torch.tensor([[0.5, 0, 0], [0.05, 0, 0], [0.0, 0.0, 0.051], [0.0, 0, 0]], device=dev)
    ids, _ = robot.find_joints(list(s5.S5_SIDE_JOINTS))
    side = SimpleNamespace(name="robot", joint_ids=ids)
    with redirect_stdout(io.StringIO()):
        r = s5.arm_side_sway_walking(env, side).cpu()
    ref = ref_arm_motion(env, side, "base_velocity", 0.05).cpu()
    check(torch.allclose(r, ref), f"== arm_motion_walking on joints {[JOINTS[i] for i in ids]}: {[round(x, 4) for x in r.tolist()]}")
    check(float(r[1]) == 0.0 and float(r[3]) == 0.0 and float(r[0]) > 0 and float(r[2]) > 0, "|cmd| 0.05 and 0 pay 0, 0.5 and 0.051 pay")
    v = robot.data.joint_vel[0, ids].cpu()
    check(abs(float(r[0]) - float((v * v).sum())) < 1e-5, "row 0 = sum of the four qd^2")

    print("proof lines: printed every PROOF_FAST ticks, parsable, no crash on all-walking / all-standing windows")
    env, robot, cmd = make_env(64, dev)
    robot.data.body_quat_w[:, 1] = quat_axis((0, 1, 0), d(-2.0)).to(dev)
    robot.data.joint_pos[:, iL], robot.data.joint_pos[:, iR] = -0.25, 0.10
    robot.data.joint_vel[:] = 0.3
    cmd[:32, 0] = 0.4
    buf = io.StringIO()
    with redirect_stdout(buf):
        for t in range(2 * s5.PROOF_FAST + 1):
            env.common_step_counter = 1000 + t
            s5.torso_pitch_bonus(env, sigma=d(3.0), shape="gauss", asset_cfg=torso)
            s5.arm_adduct_hinge(env, q_safe_left=0.2, q_safe_right=0.2, asset_cfg=cfg)
            s5.arm_side_sway_walking(env, side)
        cmd[:] = 0.0
        for t in range(s5.PROOF_FAST + 1):
            env.common_step_counter = 2000 + t
            s5.torso_pitch_bonus(env, sigma=d(3.0), shape="gauss", asset_cfg=torso)
            s5.arm_adduct_hinge(env, q_safe_left=0.2, q_safe_right=0.2, asset_cfg=cfg)
            s5.arm_side_sway_walking(env, side)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("[s5]") and "window" in ln]
    print("   " + "\n   ".join(ln[:230] for ln in lines[:3]))
    # ticks 1000..1480 -> lines at 1240 and 1480; the jump to 2000 is past the next due tick (1720) -> a 1-step line at 2000, then 2240
    check(all(len([ln for ln in lines if t in ln]) == 4 for t in ("torso_pitch", "arm_adduct", "arm_side_sway")), f"4 windows per term ({len(lines)} lines)")
    check("walk: no samples" in [ln for ln in lines if "torso_pitch" in ln][-1] and "walking 0.0 %" in [ln for ln in lines if "arm_side_sway" in ln][-1],
          "an all-standing window prints 'walk: no samples' / 'walking 0.0 %' instead of dividing by zero")
    tp = next(ln for ln in lines if "torso_pitch" in ln)
    check("pitch mean -2.00 sd 0.00" in tp and "earned 0.641" in tp, "walking pitch -2.00 deg, earned exp(-(2/3)^2) = 0.641 at sigma 3")
    aa = next(ln for ln in lines if "arm_adduct" in ln)
    check("hinge mean 0.05000" in aa and "L q_in mean +0.250" in aa and "past q_safe 100.00 %" in aa, "hinge 0.05 = (0.25 - 0.20); L past 100 %")
    print("ALL PASS" if not FAILS else f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
