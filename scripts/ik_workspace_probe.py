"""Headless probe of IKArmPoseCommand workspace coverage (no renderer needed).

Runs the Balance-IK env with zero actions for a while and, at every command
resample, records the PREVIOUS hold's outcome per hand: Cartesian target
(torso frame), final hand position, converged-or-stalled. Prints coverage
stats and saves a 3-view scatter (reached green / stalled red) so the
workspace box can be judged without the (segfaulting) GUI.

Run:
    PYTHONPATH=$PWD/source/unitree_rl_lab <isaacsim python> scripts/ik_workspace_probe.py \
        --num_envs 64 --steps 3000
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("IK workspace probe")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--out", type=str, default="ik_workspace_probe.png")
parser.add_argument(
    "--policy", type=str,
    default="/home/aspired-comp-2/Projects/robot_projects/repos/unitree_rl_lab/logs/milestones/"
            "p10_sym_scratch_12k_2026-07-01/exported/policy.pt",
    help="TorchScript balance policy to keep the robot standing (zero actions = it falls over "
         "and every IK target reads unreachable).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app = AppLauncher(args).app

import gymnasium as gym
import torch

import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg("Unitree-H1_2-Balance-IK", num_envs=args.num_envs)
    env = gym.make("Unitree-H1_2-Balance-IK", cfg=env_cfg)
    obs, _ = env.reset()

    policy = torch.jit.load(args.policy, map_location=env.unwrapped.device)
    policy.eval()
    print(f"[probe] policy loaded: {args.policy}", flush=True)

    cmd = env.unwrapped.command_manager.get_term("arm_pose_command")

    records = {"left": [], "right": []}  # (target_b, final_ee_b, converged, was_standing)
    prev = None

    for i in range(args.steps):
        # Snapshot BEFORE stepping so a resample this step lets us log the
        # finished hold with its final hand position.
        snap = {
            side: (
                cmd.target_pos_b[side].clone(),
                cmd._ee_state_b(side)[0].clone(),
                cmd.frozen[side].clone(),
                cmd.default_mode.clone(),
                cmd.time_since_resample.clone(),
            )
            for side in ("left", "right")
        }
        base_h = env.unwrapped.scene["robot"].data.root_pos_w[:, 2].clone()
        with torch.no_grad():
            actions = policy(obs["policy"])
        obs, *_ = env.step(actions)
        if prev is not None:
            resampled = cmd.time_since_resample < prev  # timer went backwards -> resample happened
            ids = torch.nonzero(resampled).flatten()
            for side in ("left", "right"):
                tgt, ee, frz, dfl, tsr = snap[side]
                for e in ids.tolist():
                    if dfl[e]:
                        continue  # peace-time hold, no Cartesian target
                    if tsr[e] < 5.0:
                        continue  # hold too short to judge reachability
                    err = torch.norm(tgt[e] - ee[e]).item()
                    records[side].append((tgt[e].cpu(), ee[e].cpu(), err < 0.05, base_h[e].item() > 0.7))
        prev = cmd.time_since_resample.clone()

    print("\n================ IK workspace probe ================", flush=True)
    for side in ("left", "right"):
        allrec = records[side]
        fallen = sum(1 for r in allrec if not r[3])
        rec = [r[:3] for r in allrec if r[3]]  # standing-only
        print(f"{side}: {len(allrec)} full-length holds, {fallen} on fallen robots (excluded)", flush=True)
        if not rec:
            print(f"{side}: no standing holds recorded", flush=True)
            continue
        n = len(rec)
        conv = sum(1 for _, _, c in rec if c)
        errs = torch.tensor([torch.norm(t - e).item() for t, e, _ in rec])
        print(f"{side}: {n} holds | reached (<5cm): {conv} ({100 * conv / n:.0f}%) | "
              f"residual err mean {errs.mean():.3f} m, p90 {errs.quantile(0.9):.3f} m", flush=True)
        stalled = [t for t, _, c in rec if not c]
        if stalled:
            st = torch.stack(stalled)
            print(f"  stalled-target centroid (torso frame): "
                  f"x {st[:, 0].mean():.2f}, y {st[:, 1].mean():.2f}, z {st[:, 2].mean():.2f}", flush=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        planes = [("x", "z", 0, 2), ("y", "z", 1, 2), ("x", "y", 0, 1)]
        for row, side in enumerate(("left", "right")):
            rec = [r[:3] for r in records[side] if r[3]]  # standing-only
            if not rec:
                continue
            tgts = torch.stack([t for t, _, _ in rec])
            ok = torch.tensor([c for _, _, c in rec])
            for col, (nx, ny, ix, iy) in enumerate(planes):
                ax = axes[row][col]
                ax.scatter(tgts[ok, ix], tgts[ok, iy], s=6, c="green", label="reached")
                ax.scatter(tgts[~ok, ix], tgts[~ok, iy], s=6, c="red", label="stalled")
                ax.set_xlabel(f"{nx} (m, torso frame)")
                ax.set_ylabel(f"{ny} (m)")
                ax.set_title(f"{side} arm — {nx}{ny}")
                ax.legend(fontsize=7)
                ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out, dpi=110)
        print(f"scatter saved: {args.out}", flush=True)
    except Exception as e:
        print(f"(scatter skipped: {e})", flush=True)

    env.close()


main()
app.close()
