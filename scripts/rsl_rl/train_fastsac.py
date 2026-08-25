"""FastSAC pilot trainer (aspired/fastsac, 2026-08-25).

Trains a Unitree task with the holosoma FastSAC port instead of rsl_rl PPO.
Mirrors train.py's env construction but drives the RAW ManagerBasedRLEnv —
no RslRlVecEnvWrapper, no OnPolicyRunner.

Run A contract: our task, our reward ledger, FastSAC as the only variable.

    python scripts/rsl_rl/train_fastsac.py --task Unitree-H1_2-LM5-Q \
        --num_envs 4096 --max_iterations 50000 --headless [--no-wandb]

VRAM note: replay buffer = num_envs x 1024 x (2*obs + 2*critic_obs + act + 4)
floats — at 4096 envs / ~80-dim obs this is ~6 GB; 8192 envs roughly doubles it.
"""

import pathlib
import sys

import gymnasium as gym

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with FastSAC.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Unitree-H1_2-LM5-Q", choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=50000,
                    help="FastSAC env steps (= gradient rounds; NOT PPO iterations).")
parser.add_argument("--run_name", type=str, default="", help="Suffix for the log dir name.")
parser.add_argument("--resume_path", type=str, default=None, help="FastSAC checkpoint (.pt) to resume from.")
parser.add_argument("--no-wandb", dest="no_wandb", action="store_true", default=False,
                    help="Disable wandb (MANDATORY for smokes).")
parser.add_argument("--no-compile", dest="no_compile", action="store_true", default=False,
                    help="Disable torch.compile (faster startup for smokes).")
parser.add_argument("--batch_size", type=int, default=None, help="Override global batch size (default 8192).")
parser.add_argument("--buffer_size", type=int, default=None, help="Override per-env buffer size (default 1024).")
parser.add_argument("--num_updates", type=int, default=None, help="Override UTD updates per step (default 8).")
parser.add_argument("--save_interval", type=int, default=None, help="Override checkpoint interval (default 1000).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import shutil
import inspect
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import parse_env_cfg

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.agents.fastsac import FastSACConfig, FastSACRunner
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    cfg = FastSACConfig()
    cfg.num_learning_iterations = args_cli.max_iterations
    cfg.compile = not args_cli.no_compile
    if args_cli.batch_size is not None:
        cfg.batch_size = args_cli.batch_size
    if args_cli.buffer_size is not None:
        cfg.buffer_size = args_cli.buffer_size
    if args_cli.num_updates is not None:
        cfg.num_updates = args_cli.num_updates
    if args_cli.save_interval is not None:
        cfg.save_interval = args_cli.save_interval

    log_root_path = os.path.abspath(os.path.join("logs", "fastsac", args_cli.task.lower()))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.run_name:
        log_dir += f"_{args_cli.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped  # ManagerBasedRLEnv — FastSAC drives it directly

    writer = SummaryWriter(log_dir=log_dir)
    if not args_cli.no_wandb:
        import wandb

        wandb.save = lambda *a, **k: None  # same no-upload policy as train.py
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "unitree_rl_lab"),
            name=f"fastsac_{os.path.basename(log_dir)}",
            dir=log_dir,
            sync_tensorboard=True,  # our runner logs via the tb writer only
            config={"task": args_cli.task, "num_envs": args_cli.num_envs,
                    **{k: getattr(cfg, k) for k in vars(cfg)}},
        )

    runner = FastSACRunner(raw_env, cfg, log_dir=log_dir, writer=writer)
    if args_cli.resume_path:
        runner.load(args_cli.resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"),
              {"algorithm": "fastsac", **{k: getattr(cfg, k) for k in vars(cfg)}})
    export_deploy_cfg(raw_env, log_dir)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    runner.learn()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
