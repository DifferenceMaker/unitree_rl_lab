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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, wrap_to_pi, yaw_quat

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


def track_vy_err_l2(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(vy_cmd - vy)^2 alone, body frame (study s2, 2026-11-13) -- track_yaw_err_l2's twin for the lateral axis. s1_pilot: 2 of 36
    trials reach walks_vy >= 0.9; track_err lumps the lateral error with the forward and yaw errors, and on a +-0.3 m/s axis next to
    a 1.2 m/s one the forward term dominates the sum. This gives sidestepping its own slope: d/de (w e^2) = 2 w e, at e = 0.1 m/s
    and w = -10 that is -2.0 per m/s. Not gated (a price on an error of the COMMANDED axis; at a zero vy command it prices lateral
    drift, which is wanted)."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return torch.square(cmd[:, 1] - asset.data.root_lin_vel_b[:, 1])


def track_vx_err_l2(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """(vx_cmd - vx)^2 alone, body frame (study s2, 2026-11-13). With track_vy_err_l2 and track_yaw_err_l2 it SPLITS track_err
    (track_vel_err_l2 = the sum of the three) into one price per axis -- operator: "isn't this the same problem we had with
    torso_stability_bonus? When we diverged [it] into two terms it was more fruitful. Shouldn't we do the same with vx, vy and wz".
    At equal weights w the three terms pay exactly what track_err paid at w."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return torch.square(cmd[:, 0] - asset.data.root_lin_vel_b[:, 0])


def track_lin_err_l1(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """|v_xy cmd - v_xy| (the length of the planar velocity error, body frame), study s2 2026-11-13. WHY an absolute-error price:
    the s1 top trials track above a threshold and STAND below it (vx / vy +-0.05 and +-0.10 measured 0.00). A squared price has slope
    2 w e, which vanishes exactly there: at e = 0.10 m/s and w = -20 it costs 0.2 /s. This one has the same slope at every error
    size: at w = -10 the same standing error costs 1.0 /s, comparable with what standing collects from the kernel (2.0-4.7 /s at the
    s1 floors) and the torso stillness bonus (1.1 /s). Ungated price on the commanded axes."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return torch.linalg.norm(cmd[:, :2] - asset.data.root_lin_vel_b[:, :2], dim=1)


def track_yaw_err_l1(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """|wz_cmd - wz|, study s2 2026-11-13: track_lin_err_l1's twin for the turn. The s1 top trials do not turn below ~0.3 rad/s
    (measured 0.00-0.05 at a 0.20-0.30 command); track_yaw's squared price at e = 0.2 and w = -12 is 0.48 /s, this one at w = -10
    is 2.0 /s."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    return torch.abs(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])


def torso_stability_lin_rel(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    std_lin: float = 0.15,
    command_name: str = "base_velocity",
    lin_rel: float = 1.0,
) -> torch.Tensor:
    """exp(-|v_torso_xy - lin_rel x v_cmd_xy|^2 / std_lin^2): torso_stability_lin with the torso's planar velocity scored RELATIVE to
    the commanded velocity (study s2, 2026-11-13). The parent's term (rewards.torso_stability_bonus, std_lin .15, std_ang 1e6) scores
    the ABSOLUTE torso speed -- its command_name only makes the YAW rate relative -- so a perfect 0.10 m/s walk earns exp(-(0.10 /
    0.15)^2) = 0.64 of the weight 3 and standing earns all of it: 1.1 /s paid for standing under that command. lin_rel = 1 pays a
    smooth walk at the commanded speed in full and still prices sway and wobble about it (operator: "we don't want to sacrifice torso
    stillness"); lin_rel = 0 is the parent's term. The command is in the yaw frame, the torso velocity in the world frame: rotate."""
    asset = env.scene[asset_cfg.name]
    v_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].squeeze(1)
    cmd = env.command_manager.get_command(command_name)
    c_b = torch.cat([cmd[:, :2], torch.zeros_like(cmd[:, :1])], dim=1)
    c_w = quat_apply(yaw_quat(asset.data.root_quat_w), c_b)[:, :2]
    return torch.exp(-torch.sum(torch.square(v_w - float(lin_rel) * c_w), dim=1) / (std_lin ** 2))


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


def foot_clearance_swing_along_cmd(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str = "base_velocity",
    cmd_thr: float = 0.03,
    lin_thr: float = 0.05,
) -> torch.Tensor:
    """lm7l 'arch' (operator 2026-11-05: "the legs get raised upwards, instead of upwards and forward if walking forward or upwards and
    backwards when walking backwards ... reward it for following the vector of velocity with feet ... a nice, natural arch to the next step
    ... we don't want the arch to be too excessive and lunge forward"): foot_clearance_swing_cmd with its speed factor tanh(tanh_mult |v_xy|)
    -- which pays a swing in ANY planar direction, so the outward arcs earned like a straight stride -- replaced by tanh(tanh_mult x
    max(0, v_foot . u)), u = the commanded planar velocity direction in the base frame: a foot moving along the command earns the lift
    kernel, a foot moving backward or sideways against it earns nothing. tanh saturates at ~1 m/s of along-command speed, so a lunge pays
    no more than a stride (the cap the operator asked for; the step-length prices stay). Under a pure turn (|v_cmd_xy| < lin_thr) u is
    undefined: the old speed factor applies, so the turn's clearance is unchanged. One term, not two -- the modified swing reward IS the
    arch reward (no competition between a lift term and a direction term). Command-gated as before (rule 32)."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    z = asset.data.body_pos_w[:, ids, 2]
    v_w = asset.data.body_lin_vel_w[:, ids, :3]
    n = v_w.shape[1]
    q = asset.data.root_quat_w.unsqueeze(1).expand(-1, n, -1).reshape(-1, 4)
    v_b = quat_apply_inverse(q, v_w.reshape(-1, 3)).reshape(-1, n, 3)[..., :2]      # planar foot velocity in the base frame
    cmd = env.command_manager.get_command(command_name)
    cmd_xy = cmd[:, :2]
    cmd_n = torch.linalg.norm(cmd_xy, dim=1)
    u = cmd_xy / cmd_n.clamp(min=1e-6).unsqueeze(1)
    along = torch.sum(v_b * u.unsqueeze(1), dim=2).clamp(min=0.0)                      # (envs, feet) speed along the command
    speed = torch.linalg.norm(v_b, dim=2)
    has_dir = (cmd_n > lin_thr).unsqueeze(1)
    factor = torch.tanh(tanh_mult * torch.where(has_dir, along, speed))
    per_foot = factor * torch.exp(-torch.square(z - target_height) / std)
    gate = torch.linalg.norm(cmd, dim=1) > cmd_thr
    return per_foot.max(dim=1).values * gate.float()
