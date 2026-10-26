"""LM7U terms (the lm7b_u wave, 2026-10-09). Own module on purpose: the pod's mdp/__init__.py differs from this checkout's, so new
terms arrive as a new file + one appended import line.

Energy for a walker (operator: "On the walker the same price taxes walking itself, so standing becomes the cheapest way to be still.
Precisely."): a flat power price stopped the walk three times (lm6c energy rows, lm7b_energy 55 -> 20 %, lm7_union_energy 58 -> 20 %).
  energy_cmd_gated     full price at a zero command, cut to `moving_scale` while a velocity is commanded
  energy_per_distance  cost of transport on the COMMANDED speed, normalised to the plain price at a stand
Rotation (probe_rotation.py, 2026-10-09: under a 0.5 rad/s spin command the walkers STAND and earn 6.2/s more than at zero command):
  foot_clearance_swing_cmd  clearance that pays only for a MOVING foot (the old form is maximal for a motionless one)
  track_yaw_err_l2          a yaw-only L2, the slope the flat angular kernel does not give
Precision (operator: "when the policy has learned to track with 0.1 or 0.15 ... decrease the std band by 0.05 ... so it has levels"):
  tracking_std_levels       curriculum: tighten both tracking kernels one level when the tracking income passes gate_frac x weight
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _joint_power(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """sum |qvel| x |applied torque| -- the quantity mdp.energy prices."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) * torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=-1)


def energy_cmd_gated(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    moving_scale: float = 0.5,
    cmd_ref: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Power, priced in full at a zero command and cut to `moving_scale` of it once a velocity is commanded (operator: "keep the
    energy as it was for 0 commanded velocity but cut it by 2x or 4x when velocity is commanded"). The gate fades linearly over
    |cmd| in [0, cmd_ref] on the FULL command norm (vx, vy, wz) -- the trunk's standing threshold -- so a turn counts as moving and
    slow-band commands are not charged the standing price."""
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    s = (cmd / cmd_ref).clamp(max=1.0)
    return _joint_power(env, asset_cfg) * (1.0 - (1.0 - moving_scale) * s)


def energy_per_distance(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    v_floor: float = 0.25,
    wz_lever: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Cost of transport on the commanded speed: P x v_floor / max(|v_xy cmd| + wz_lever |wz cmd|, v_floor) = CoT x m g v_floor.
    At a stand (and up to v_floor) the price per watt equals plain `energy` at the same weight; at 0.5 m/s it is halved, at 1.0 m/s
    quartered -- a price per metre walked, not per second. The COMMANDED speed is observable and cannot be gamed by drifting;
    turning counts through wz_lever (wz 0.5 rad/s = the 0.25 m/s floor)."""
    c = env.command_manager.get_command(command_name)
    speed = torch.linalg.norm(c[:, :2], dim=1) + wz_lever * c[:, 2].abs()
    return _joint_power(env, asset_cfg) * v_floor / speed.clamp(min=v_floor)


def foot_clearance_swing_cmd(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str = "base_velocity",
    cmd_thr: float = 0.1,   # lm7d: the walking gate, was a hard-coded 0.1
) -> torch.Tensor:
    """Clearance that pays only for a MOVING foot: max over feet of tanh(tanh_mult |v_foot_xy|) x exp(-(z - target)^2 / std),
    command-gated (|cmd| > 0.1). foot_clearance_reward = exp(-sum (z - target)^2 tanh(|v|) / std) is MAXIMAL for a motionless foot,
    so under any nonzero command standing still collected 11.85 of 12 (probe_rotation 2026-10-09) -- a disobedience subsidy under a
    turn command and under slow linear commands alike. Here a planted foot pays 0 and a foot swinging near the target height pays ~1."""
    asset = env.scene[asset_cfg.name]
    z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    v = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    per_foot = torch.tanh(tanh_mult * v) * torch.exp(-torch.square(z - target_height) / std)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    return per_foot.max(dim=1).values * (cmd > cmd_thr).float()


def track_yaw_err_l2(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(wz_cmd - wz)^2 alone. The angular kernel (std .05) pays 0.06 of 14 at 0.15-0.30 rad/s against a 0.5 command, and track_err
    lumps yaw with the linear error; this gives the turn its own slope from standstill."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return torch.square(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])


def tracking_std_levels(
    env: "ManagerBasedRLEnv",
    env_ids,
    reward_terms: tuple = ("track_lin_vel_xy", "track_ang_vel_z"),
    levels: tuple = (0.15, 0.10, 0.05),
    gate_term: str = "track_lin_vel_xy",
    gate_frac: float = 0.5,
    hold_until_step: int = 0,
) -> torch.Tensor:
    """Tracking-kernel ladder, the trigger of lin_vel_cmd_levels: at an episode boundary, if the gate term's mean episode income per
    second of the resetting envs exceeds gate_frac x its weight, both tracking kernels tighten one level; never back. Performance-gated,
    not step-based, so a warmstart (--resume_curriculum) starts at levels[0]. Returns the current std for the Curriculum/ log.
    hold_until_step (lm7f pc_soft, 2026-10-26): no climb while env.common_step_counter < hold_until_step -- with --resume_curriculum the
    counter is the parent's iteration x 24, so (parent_it + N) x 24 holds the wide kernel for the first N iterations of a warmstart (the
    padded-critic rows climbed to .05 inside the 300-it warm-up and fell out of the tracking basin for good when the actor unfroze)."""
    rm = env.reward_manager
    if not hasattr(env, "_track_std_level"):
        env._track_std_level = 0
        for n in reward_terms:
            rm.get_term_cfg(n).params["std"] = levels[0]
    n_ids = env.num_envs if isinstance(env_ids, slice) else len(env_ids)   # the resume path calls compute(env_ids=None) -> slice
    if (env._track_std_level < len(levels) - 1 and n_ids > 0 and env.common_step_counter % env.max_episode_length == 0
            and env.common_step_counter >= hold_until_step):
        income = torch.mean(rm._episode_sums[gate_term][env_ids]) / env.max_episode_length_s
        if income > rm.get_term_cfg(gate_term).weight * gate_frac:
            env._track_std_level += 1
            for n in reward_terms:
                rm.get_term_cfg(n).params["std"] = levels[env._track_std_level]
    return torch.tensor(levels[env._track_std_level], device=env.device)


# ── lm7d (2026-10-19) ──────────────────────────────────────────────────────────────────────────
def body_ang_vel_l2_walking(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    cmd_thr: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
) -> torch.Tensor:
    """The torso's angular-velocity L2 while a command is ACTIVE (|cmd| >= cmd_thr) -- the complement of
    body_ang_vel_l2_standing. The stillness terms are all standing-gated, so a slowly walking torso is priced by
    nothing (operator 2026-10-19: "total control over the stability of the torso while walking at low speeds").
    Meant at a LOW weight next to the standing term, never instead of it."""
    asset = env.scene[asset_cfg.name]
    w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    return torch.sum(w * w, dim=(1, 2)) * (cmd >= cmd_thr).float()


def feet_yaw_l2_turn_gated(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    wz_ref: float = 0.5,
) -> torch.Tensor:
    """Sum over the feet of (foot yaw - base yaw)^2, faded out with the commanded yaw rate (x (1 - min(|wz|/wz_ref, 1)),
    the joint_deviation_l1_turn_gated fade). Foot ORIENTATION has never been priced (stance_bonus-kernel-is-flat): the
    outward-rotated leg, the "right leg forward and rotated outward" default stance and the both-legs-outward stance are all
    free. Parallel feet under the heading; a commanded turn may rotate them."""
    asset = env.scene[asset_cfg.name]
    base_yaw = euler_xyz_from_quat(asset.data.root_quat_w)[2]
    fq = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    n_feet = fq.shape[1]
    foot_yaw = euler_xyz_from_quat(fq.reshape(-1, 4))[2].reshape(-1, n_feet)
    d = wrap_to_pi(foot_yaw - base_yaw.unsqueeze(1))
    wz = env.command_manager.get_command(command_name)[:, 2].abs()
    gate = 1.0 - (wz / wz_ref).clamp(max=1.0)
    return torch.sum(d * d, dim=1) * gate
