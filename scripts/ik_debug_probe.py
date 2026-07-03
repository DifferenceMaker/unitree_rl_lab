"""Minimal debug probe for IKArmPoseCommand: is the anchor right, do the
joint targets move, does the Cartesian error shrink? Prints flushed."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("IK debug probe")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app = AppLauncher(args).app

import gymnasium as gym
import torch

import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def p(*a):
    print(*a, flush=True)


def main():
    env_cfg = parse_env_cfg("Unitree-H1_2-Balance-IK", num_envs=args.num_envs)
    env_cfg.terminations.base_height = None
    env_cfg.events.push_robot = None
    env_cfg.events.sustained_push_apply = None
    env_cfg.curriculum.push_velocity = None
    env_cfg.curriculum.sustained_push = None
    env = gym.make("Unitree-H1_2-Balance-IK", cfg=env_cfg)
    obs, _ = env.reset()

    u = env.unwrapped
    cmd = u.command_manager.get_term("arm_pose_command")
    robot = u.scene["robot"]
    policy = torch.jit.load(
        "/home/aspired-comp-2/Projects/robot_projects/repos/unitree_rl_lab/logs/milestones/"
        "p10_sym_scratch_12k_2026-07-01/exported/policy.pt", map_location=u.device)
    policy.eval()

    for i in range(args.steps):
        with torch.no_grad():
            act = policy(obs["policy"])
        obs, *_ = env.step(act)
        if i == 1:
            p("anchor default_ee_pos_b left:", cmd.default_ee_pos_b["left"][0].tolist())
            p("anchor default_ee_pos_b right:", cmd.default_ee_pos_b["right"][0].tolist())
            p("default_mode[0]:", bool(cmd.default_mode[0]))
        if i % 20 == 0:
            side = "left"
            ee_b, _, _ = cmd._ee_state_b(side)
            jt = cmd.joint_targets[0, cmd.arm_cols[side]]
            jp = robot.data.joint_pos[0, cmd.arm_jids[side]]
            err = torch.norm(cmd.target_pos_b[side][0] - ee_b[0]).item()
            p(f"step {i:4d} | tgt_b {cmd.target_pos_b[side][0].tolist()}"
              f" | ee_b {[round(v, 3) for v in ee_b[0].tolist()]}"
              f" | err {err:.3f} | frozen {bool(cmd.frozen[side][0])}"
              f" | t_resample {cmd.time_since_resample[0].item():.1f}")
            p(f"          jt {[round(v, 2) for v in jt.tolist()]}")
            p(f"          jp {[round(v, 2) for v in jp.tolist()]}")
            p(f"          base_h {robot.data.root_pos_w[0, 2].item():.3f} default_mode {bool(cmd.default_mode[0])}")
        if i == 200:
            base_h = robot.data.root_pos_w[0, 2].item()
            p(f"base height env0 at step 200: {base_h:.3f}")

    env.close()


main()
app.close()
