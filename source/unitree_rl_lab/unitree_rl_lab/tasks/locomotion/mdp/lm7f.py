"""LM7F terms (2026-10-26): the lm7e sim2sim keepers composed on lm7e_A_kit, plus the policy-activation DR.

Own module (the lm7u / lm7e convention: a new file + one appended import line).

Operator 2026-10-26: "we really need to start straining our policies with the policy activation because not always the policy handover
will be perfect" and "My plan is to never start from FixStand but from our own balance policy. Both will need to be trained to take
handover from each other." Two forms of that DR live here: v1 = wider blind offsets at reset (plain Isaac events, see lm7f_env_cfg.
gene_activation), v2 = `reset_from_pose_bank`: a share of the episodes starts from a (q, qd, tilt, velocity) sample RECORDED from the
other policy standing in Isaac (`aspired-isaac-lab/scripts/tools/record_pose_bank.py`), so the walk policy's first step is the
balance policy's last.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from .lm7e import _cmd, _foot_rpy, _turn_fade

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# ── foot yaw bonus, split by command ─────────────────────────────────────────────────────────────────────────────────────────
def feet_yaw_bonus_split(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    sigma_deg: float = 3.0,
    wz_ref: float = 0.5,
    cmd_thr: float = 0.03,
    cmd_scale: float = 0.5,
) -> torch.Tensor:
    """lm7e feet_yaw_bonus_turn_faded x (1 at zero command, cmd_scale under a command): with weight 6 and cmd_scale .5 the
    parallel-feet income is 6 while standing and the lm7e 3 while walking (operator: amp it where it matters, leave the walking
    toe-out alone). The command is observed, so the gate is legal (RL_BIBLE: never gate on unobservable state)."""
    _, _, y = _foot_rpy(env, asset_cfg)
    sigma = sigma_deg * torch.pi / 180.0
    base = torch.exp(-y.abs().sum(dim=1) / sigma) * _turn_fade(env, command_name, wz_ref)
    moving = _cmd(env, command_name)[:, :3].norm(dim=1) > cmd_thr
    return base * torch.where(moving, torch.full_like(base, cmd_scale), torch.ones_like(base))


# ── policy-activation DR v2: reset from a recorded pose bank ─────────────────────────────────────────────────────────────────
_BANKS: dict = {}


def _bank_path(bank: str) -> str:
    if os.path.isabs(bank):
        return bank
    root = os.environ.get("ROBOT_ASSETS_DIR")
    if not root:
        raise RuntimeError("[lm7f] ROBOT_ASSETS_DIR is not set; the pose bank lives in aspired-isaac-lab/assets/pose_banks/")
    return os.path.join(root, "pose_banks", bank)


def _load_bank(env, bank: str, asset) -> dict:
    key = (bank, str(env.device))
    if key in _BANKS:
        return _BANKS[key]
    path = _bank_path(bank)
    d = torch.load(path, map_location="cpu", weights_only=False)
    names = list(d["joint_names"])
    env_names = list(asset.joint_names)
    col = torch.full((len(env_names),), -1, dtype=torch.long)
    for c, n in enumerate(names):
        if n in env_names:
            col[env_names.index(n)] = c
    B = {k: torch.as_tensor(d[k]).float().to(env.device) for k in ("q", "qd", "roll_pitch", "root_z", "lin_vel_b", "ang_vel_b")}
    B["col"] = col.to(env.device)
    B["n"] = int(B["q"].shape[0])
    missing = [env_names[j] for j in range(len(env_names)) if col[j] < 0]
    print(f"[lm7f] pose bank {path}: {B['n']} samples x {len(names)} joints from {d.get('source', '?')}; "
          f"env joints without a bank column: {missing or 'none'}")
    _BANKS[key] = B
    return B


def reset_from_pose_bank(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    bank: str,
    prob: float = 0.3,
    pos_noise: float = 0.03,
    vel_noise: float = 0.3,
    xy_range: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """A share `prob` of the resetting envs starts from a bank sample: joint positions + velocities (by joint NAME, + a small blind
    noise so the bank is not memorised), base height, roll and pitch from the sample, a random heading, the sample's base linear and
    angular velocity rotated into that heading. Runs after reset_base / reset_robot_joints (appended term) and overwrites their draw
    for the picked envs; the other 1 - prob keep the trunk's reset."""
    asset = env.scene[asset_cfg.name]
    B = _load_bank(env, bank, asset)
    pick = env_ids[torch.rand(len(env_ids), device=env.device) < prob]
    n = int(pick.numel())
    if n == 0:
        return pick
    rows = torch.randint(0, B["n"], (n,), device=env.device)
    col, has = B["col"], B["col"] >= 0
    q = asset.data.default_joint_pos[pick].clone()
    qd = torch.zeros_like(q)
    q[:, has] = B["q"][rows][:, col[has]]
    qd[:, has] = B["qd"][rows][:, col[has]]
    q = q + math_utils.sample_uniform(-pos_noise, pos_noise, q.shape, env.device)
    qd = qd + math_utils.sample_uniform(-vel_noise, vel_noise, qd.shape, env.device)
    lim = asset.data.soft_joint_pos_limits[pick]
    q = q.clamp(lim[..., 0], lim[..., 1])
    vlim = asset.data.soft_joint_vel_limits[pick]
    qd = qd.clamp(-vlim, vlim)
    # root pose: the trunk's xy spread around the env origin, the sample's height / roll / pitch, a random heading
    root = asset.data.default_root_state[pick].clone()
    pos = root[:, 0:3] + env.scene.env_origins[pick]
    pos[:, 0:2] += math_utils.sample_uniform(-xy_range, xy_range, (n, 2), env.device)
    pos[:, 2] = env.scene.env_origins[pick][:, 2] + B["root_z"][rows]
    rp = B["roll_pitch"][rows]
    yaw = math_utils.sample_uniform(-torch.pi, torch.pi, (n,), env.device)
    quat = math_utils.quat_from_euler_xyz(rp[:, 0], rp[:, 1], yaw)
    v_w = math_utils.quat_apply(quat, B["lin_vel_b"][rows])
    w_w = math_utils.quat_apply(quat, B["ang_vel_b"][rows])
    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=pick)
    asset.write_root_velocity_to_sim(torch.cat([v_w, w_w], dim=-1), env_ids=pick)
    asset.write_joint_state_to_sim(q, qd, env_ids=pick)
    return pick   # lm7h: the handover hold wrapper needs the picked envs (the event manager ignores a return value)
