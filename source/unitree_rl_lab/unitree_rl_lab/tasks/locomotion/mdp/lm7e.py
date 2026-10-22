"""LM7E / LM8 terms (2026-10-22): the walking-recipe survey turned into levers.

Own module on purpose (like lm7u.py): the pod's mdp/__init__.py differs from this checkout's, so new terms arrive as a new file plus one
appended import line and cherry-pick without conflict.

Sources (notes/walking_recipes_vs_lm7_2026-10-21.md): #83 "Architecture Is All You Need" (G1, real; phase-contact with a stance intent
at zero command, stand_still L1 anchor, feet_flat, swing-height target decaying near zero command), #333 ADAPT (one 0.05 gate, angular
momentum), HOMIE (feet-parallel), X-Loco (stuck), humanoid-gym (track_vel_hard, low_speed, reference swing pose, 0.64 s clock), Menlo /
Asimov (per-joint pose tolerances, air time, contact-force limit, targeted DR incl. 0-1 step observation delay, privileged critic that
sees contact).

Conventions: every reward returns (num_envs,) per step; Isaac Lab multiplies by weight x dt. "turn-faded" = x (1 - min(|wz_cmd| / wz_ref, 1)),
the lm7d feet_yaw fade. Foot bodies = the ankle_roll links; the contact sensor is the scene's `contact_forces` (history 3, air time tracked).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


# ── helpers ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _cmd(env, command_name):
    return env.command_manager.get_command(command_name)


def _turn_fade(env, command_name, wz_ref):
    wz = _cmd(env, command_name)[:, 2].abs()
    return 1.0 - (wz / wz_ref).clamp(max=1.0)


def _foot_rpy(env, asset_cfg):
    """Foot roll, pitch, yaw (each num_envs x n_feet) in the world frame, yaw relative to the base heading."""
    asset = env.scene[asset_cfg.name]
    fq = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    n = fq.shape[1]
    r, p, y = euler_xyz_from_quat(fq.reshape(-1, 4))
    base_yaw = euler_xyz_from_quat(asset.data.root_quat_w)[2]
    r = wrap_to_pi(r.reshape(-1, n)); p = wrap_to_pi(p.reshape(-1, n))
    y = wrap_to_pi(y.reshape(-1, n) - base_yaw.unsqueeze(1))
    return r, p, y


def _contact(env, sensor_cfg, thr=1.0):
    sensor: "ContactSensor" = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return f.norm(dim=-1) > thr, f


# ── A: foot orientation as INCOME (operator 2026-10-22: a reward, not a penalty) ───────────────────────────────────────────────
def feet_yaw_bonus_turn_faded(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    sigma_deg: float = 3.0,
    wz_ref: float = 0.5,
) -> torch.Tensor:
    """exp(-sum_f |yaw_f - yaw_base| / sigma): an L1 inside the exponential, so the gradient at parallel feet is 1/sigma, not zero
    (the lm7d feet_yaw L2 was flat at the bottom: 10 deg on both feet cost 0.6/s of a 66/s ledger). sigma 3 deg: both feet 3 deg out
    pay e^-2 = 0.14 of the ceiling. Faded under a commanded turn exactly like the L2 was."""
    _, _, y = _foot_rpy(env, asset_cfg)
    sigma = sigma_deg * torch.pi / 180.0
    return torch.exp(-y.abs().sum(dim=1) / sigma) * _turn_fade(env, command_name, wz_ref)


def feet_flat_bonus(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sigma_deg: float = 5.0,
) -> torch.Tensor:
    """#83 feet_flat as income: exp(-sum_f c_f (|roll_f| + |pitch_f|) / sigma) for feet IN CONTACT (a swinging foot may tilt).
    Prices the "pitches them outwards" the operator saw on the lm7d footyaw row; nothing in the ledger priced foot roll/pitch."""
    r, p, _ = _foot_rpy(env, asset_cfg)
    c, _ = _contact(env, sensor_cfg)
    sigma = sigma_deg * torch.pi / 180.0
    return torch.exp(-((r.abs() + p.abs()) * c.float()).sum(dim=1) / sigma)


# ── A: air time, angular momentum, per-joint posture, stance intent, low speed ──────────────────────────────────────────────────
def feet_air_time_touchdown(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    threshold: float = 0.25,
    cmd_thr: float = 0.03,
) -> torch.Tensor:
    """legged_gym feet_air_time: at each touchdown pay (t_air - threshold), summed over feet, only under a command. The threshold
    must sit UNDER our swing (period .75 x duty .45 = 0.34 s): legged_gym's 0.5 would punish every step; 0.25 rewards a full swing
    by +0.09 and a shuffle (0.1 s) by -0.15. Menlo: +0.5 'encourages dynamic gaits rather than shuffling'."""
    sensor: "ContactSensor" = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    r = torch.sum((last_air - threshold).clamp(min=-0.5, max=0.5) * first_contact, dim=1)
    return r * (torch.norm(_cmd(env, command_name)[:, :2], dim=1) > cmd_thr)


def angular_momentum_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """||L||^2 of the whole body about its CoM, L = sum_i m_i (p_i - p_com) x (v_i - v_com) (#333 -0.02, Menlo -0.03 relative to a
    tracking of 2). The thing the arms cancel in a human walk; our arms are held, so the legs absorb it as torso pitch."""
    asset = env.scene[asset_cfg.name]
    m = getattr(env, "_lm7e_masses", None)
    if m is None:
        m = asset.root_physx_view.get_masses().to(env.device)           # (num_envs, num_bodies)
        env._lm7e_masses = m
    p = asset.data.body_com_pos_w                                        # (N, B, 3)
    v = asset.data.body_com_lin_vel_w
    mtot = m.sum(dim=1, keepdim=True)
    p_com = (m.unsqueeze(-1) * p).sum(dim=1) / mtot
    v_com = (m.unsqueeze(-1) * v).sum(dim=1) / mtot
    L = (m.unsqueeze(-1) * torch.cross(p - p_com.unsqueeze(1), v - v_com.unsqueeze(1), dim=-1)).sum(dim=1)
    return (L * L).sum(dim=1)


def stance_bonus_vec(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    std_map: dict,
    std_default: float = 0.3,
) -> torch.Tensor:
    """stance_bonus with a PER-JOINT std (Menlo's asymmetric pose tolerances): exp(-sum_j ((q_j - q_j^def) / sigma_j)^2). std_map maps a
    joint-name substring to its sigma; joints not listed get std_default. A joint deviating by sigma_j costs e^-1."""
    asset = env.scene[asset_cfg.name]
    key = f"_lm7e_std_{id(std_map)}"
    sig = getattr(env, key, None)
    if sig is None:
        names = [asset.joint_names[i] for i in asset_cfg.joint_ids]
        vals = []
        for n in names:
            s = std_default
            for k, v in std_map.items():
                if k in n:
                    s = v
            vals.append(s)
        sig = torch.tensor(vals, device=env.device)
        setattr(env, key, sig)
    dev = (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]) / sig
    return torch.exp(-(dev * dev).sum(dim=1))


def feet_gait_stance_intent(
    env: "ManagerBasedRLEnv",
    period: float,
    offset: list,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.55,
    command_name: str = "base_velocity",
    cmd_thr: float = 0.03,
) -> torch.Tensor:
    """#83 phase-contact consistency: stance intent s_i = 1[phi_i < tau OR ||u_cmd|| < eps]; reward = sum_i 1[c_i == s_i]. Under a
    zero command both feet are 'intended stance', so double support IS the standing reward and no separate standing term is needed
    (their words). Our feet_gait instead pays 0 at a zero command, leaving standing feet to the posture kernel."""
    sensor: "ContactSensor" = env.scene.sensors[sensor_cfg.name]
    is_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    leg_phase = torch.cat([(global_phase + o) % 1.0 for o in offset], dim=-1)
    standing = (torch.norm(_cmd(env, command_name), dim=1) < cmd_thr).unsqueeze(1)
    is_stance = (leg_phase < threshold) | standing
    return (is_stance == is_contact).float().sum(dim=1)


def low_speed_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    cmd_thr: float = 0.03,
) -> torch.Tensor:
    """humanoid-gym low_speed (as a reward with weight ~0.2): forward speed vs the forward command. |v| < 0.5|cmd| -> -1 (too slow),
    |v| > 1.2|cmd| -> 0, else +1.2; opposite sign -> -2; only while |cmd_x| > cmd_thr. The slow-command fidelity term for the
    wide-kernel design (the ladder stopping at .10 needs it)."""
    asset = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_b[:, 0]
    c = _cmd(env, command_name)[:, 0]
    ratio = v.abs() / c.abs().clamp(min=1e-3)
    r = torch.where(ratio < 0.5, torch.full_like(v, -1.0), torch.where(ratio > 1.2, torch.zeros_like(v), torch.full_like(v, 1.2)))
    r = torch.where(torch.sign(v) != torch.sign(c), torch.full_like(v, -2.0), r)
    return r * (c.abs() > cmd_thr)


def foot_clearance_decay_cmd(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.10,
    foot_thickness: float = 0.02,
    kappa: float = 5.0,
    std: float = 0.05,
    tanh_mult: float = 2.0,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """#83 swing-height: the target decays to the foot thickness near a zero command, h* = h_foot + (h_target - h_foot) tanh(kappa ||v_cmd||)
    (kappa 5: 0.46 of the lift at 0.10 m/s, 0.9 at 0.3), instead of a gate that cuts the term at a threshold. Same kernel as the
    trunk's foot_clearance_reward_cmd otherwise: exp(-sum_f (z_f - h*)^2 tanh(mult |v_f,xy|) / std)."""
    asset = env.scene[asset_cfg.name]
    z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    vxy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    vc = torch.norm(_cmd(env, command_name)[:, :2], dim=1)
    h = foot_thickness + (target_height - foot_thickness) * torch.tanh(kappa * vc)
    err = (z - h.unsqueeze(1)) ** 2 * torch.tanh(tanh_mult * vxy)
    return torch.exp(-err.sum(dim=1) / std)


def joint_deviation_l1_standing(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    cmd_thr: float = 0.005,
) -> torch.Tensor:
    """#83 stand_still: sum_j |q_j - q_j^def| x 1[||u_cmd|| < eps] -- an L1 (gradient everywhere) whole-body anchor that exists only at a
    zero command. The literature's answer to our flat Gaussian posture kernel."""
    asset = env.scene[asset_cfg.name]
    dev = (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).abs().sum(dim=1)
    return dev * (torch.norm(_cmd(env, command_name), dim=1) < cmd_thr)


def track_vel_hard(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, command_name: str = "base_velocity") -> torch.Tensor:
    """humanoid-gym track_vel_hard: 0.5 (exp(-10||e_lin||) + exp(-10|e_ang|)) - 0.2 (||e_lin|| + |e_ang|) -- a sharp cusp at zero
    error next to the wide Gaussian: the precision the wide kernel lacks, with a linear slope far out."""
    asset = env.scene[asset_cfg.name]
    c = _cmd(env, command_name)
    e_lin = (c[:, :2] - asset.data.root_lin_vel_b[:, :2]).norm(dim=1)
    e_ang = (c[:, 2] - asset.data.root_ang_vel_b[:, 2]).abs()
    return 0.5 * (torch.exp(-10.0 * e_lin) + torch.exp(-10.0 * e_ang)) - 0.2 * (e_lin + e_ang)


def feet_contact_number(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    period: float = 0.64,
    threshold: float = 0.1,
) -> torch.Tensor:
    """humanoid-gym feet_contact_number: s = sin(2 pi phi); left stance s >= 0, right stance s < 0, both when |s| < threshold; per
    foot +1 when contact == stance else -0.3; mean over feet. The clock runs from reset regardless of the command (zero command =
    stepping in place, as published)."""
    sensor: "ContactSensor" = env.scene.sensors[sensor_cfg.name]
    is_contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 5.0
    phi = (env.episode_length_buf * env.step_dt) % period / period
    s = torch.sin(2 * torch.pi * phi)
    left = (s >= 0) | (s.abs() < threshold)
    right = (s < 0) | (s.abs() < threshold)
    stance = torch.stack([left, right], dim=1)
    return torch.where(is_contact == stance, torch.ones_like(s).unsqueeze(1), torch.full_like(s, -0.3).unsqueeze(1)).mean(dim=1)


def joint_pos_ref_swing(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    period: float = 0.64,
    scale: float = 0.17,
) -> torch.Tensor:
    """humanoid-gym joint_pos (their heaviest term, 1.6): the swing leg tracks a phase reference q_ref = q_def + scale x [hip_pitch 1,
    knee 2, ankle_pitch 1] x sin-magnitude, the stance leg q_def; reward exp(-2||q - q_ref||) - 0.2 clamp(||q - q_ref||, 0, 0.5).
    asset_cfg.joint_names must list, in order: left hip pitch, left knee, left ankle pitch, right hip pitch, right knee, right ankle pitch."""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    phi = (env.episode_length_buf * env.step_dt) % period / period
    s = torch.sin(2 * torch.pi * phi)
    shape = torch.tensor([1.0, 2.0, 1.0], device=env.device) * scale
    left_swing = (s < 0).float().unsqueeze(1)      # left stance when s >= 0
    right_swing = (s >= 0).float().unsqueeze(1)
    ref = q0.clone()
    ref[:, 0:3] = q0[:, 0:3] + shape * s.abs().unsqueeze(1) * left_swing
    ref[:, 3:6] = q0[:, 3:6] + shape * s.abs().unsqueeze(1) * right_swing
    d = (q - ref).norm(dim=1)
    return torch.exp(-2.0 * d) - 0.2 * d.clamp(min=0.0, max=0.5)


def stuck_penalty(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, command_name: str = "base_velocity") -> torch.Tensor:
    """X-Loco stuck: 1[||v||^2 <= 0.1 and ||c_v||^2 >= 0.2] -- standing under a real command is priced flat, whatever else pays."""
    asset = env.scene[asset_cfg.name]
    v2 = (asset.data.root_lin_vel_b[:, :2] ** 2).sum(dim=1)
    c2 = (_cmd(env, command_name)[:, :2] ** 2).sum(dim=1)
    return ((v2 <= 0.1) & (c2 >= 0.2)).float()


# ── B: observation delay (Menlo: 0-2 steps per CAN group; ours 0-1 control step, drawn per episode) ───────────────────────────
def _delayed(env, key, x, max_delay):
    """Return x delayed by a per-env integer 0..max_delay (drawn at each episode start). A ring of max_delay+1 samples lives on env."""
    buf = getattr(env, key, None)
    new_ep = env.episode_length_buf == 0
    if buf is None or buf["hist"].shape[1:] != x.shape:
        hist = x.unsqueeze(0).repeat(max_delay + 1, 1, 1).clone()
        d = torch.randint(0, max_delay + 1, (x.shape[0],), device=x.device)
        buf = {"hist": hist, "d": d}
        setattr(env, key, buf)
    hist, d = buf["hist"], buf["d"]
    hist = torch.cat([x.unsqueeze(0), hist[:-1]], dim=0)
    if new_ep.any():
        hist[:, new_ep] = x[new_ep].unsqueeze(0)
        d[new_ep] = torch.randint(0, max_delay + 1, (int(new_ep.sum()),), device=x.device)
    buf["hist"] = hist
    idx = torch.arange(x.shape[0], device=x.device)
    return hist[d, idx]


def joint_pos_rel_delayed(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), max_delay: int = 1) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    x = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return _delayed(env, "_lm7e_delay_qpos", x, max_delay)


def joint_vel_rel_delayed(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), max_delay: int = 1) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    x = asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    return _delayed(env, "_lm7e_delay_qvel", x, max_delay)


# ── C: privileged critic (Menlo: 'By exposing toe state to the critic, the policy learns the relationship between toe behavior and
#      stability'; contact forces the same) -- appended LAST in the critic group, the actor untouched ─────────────────────────────
def feet_contact_flags(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, threshold: float = 5.0) -> torch.Tensor:
    c, _ = _contact(env, sensor_cfg, threshold)
    return c.float()


def feet_force_z(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, scale_n: float = 700.0) -> torch.Tensor:
    """Vertical ground reaction per foot in units of ~one body weight (H1-2 ~ 70 kg)."""
    _, f = _contact(env, sensor_cfg)
    return f[:, :, 2] / scale_n


def feet_height(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids, 2]


def feet_xy_speed(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)


def feet_air_time_obs(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: "ContactSensor" = env.scene.sensors[sensor_cfg.name]
    return sensor.data.current_air_time[:, sensor_cfg.body_ids].clamp(max=1.0)


# ── D: terms the recipe transplants need and the trunk does not have ───────────────────────────────────────────────────────────
def joint_deviation_l2(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """unitree_rl_gym hip_pos: sum_j (q_j - q_j^def)^2 over the selected joints (their hip yaw + roll)."""
    asset = env.scene[asset_cfg.name]
    d = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return (d * d).sum(dim=1)


def feet_swing_height_penalty(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, target: float = 0.08
) -> torch.Tensor:
    """unitree_rl_gym feet_swing_height: sum over feet NOT in contact of (z_f - target)^2 (their -20 at 0.08 m)."""
    asset = env.scene[asset_cfg.name]
    z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    c, _ = _contact(env, sensor_cfg)
    return (((z - target) ** 2) * (~c).float()).sum(dim=1)


def upright_bonus_exp(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, sigma_sq: float = 0.2) -> torch.Tensor:
    """#333 / Menlo upright: exp(-||g_xy||^2 / sigma^2) on the projected gravity, ungated (every command)."""
    asset = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b[:, :2]
    return torch.exp(-(g * g).sum(dim=1) / sigma_sq)


def hip_wall_bonus(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, all_cfg: SceneEntityCfg, deadband: float = 0.1) -> torch.Tensor:
    """humanoid-gym default_joint_pos: exp(-100 clamp(sum |hip roll, yaw dev| - deadband, 0, 50)) - 0.01 ||q - q_def|| (all joints).
    A wall on hip rotation beyond 0.1 rad, flat inside it."""
    asset = env.scene[asset_cfg.name]
    hip = (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).abs().sum(dim=1)
    allq = (asset.data.joint_pos[:, all_cfg.joint_ids] - asset.data.default_joint_pos[:, all_cfg.joint_ids]).norm(dim=1)
    return torch.exp(-100.0 * (hip - deadband).clamp(min=0.0, max=50.0)) - 0.01 * allq


def vel_mismatch_exp(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """humanoid-gym vel_mismatch_exp: (exp(-10 v_z^2) + exp(-5 ||w_xy||)) / 2."""
    asset = env.scene[asset_cfg.name]
    vz = asset.data.root_lin_vel_b[:, 2]
    wxy = asset.data.root_ang_vel_b[:, :2].norm(dim=1)
    return 0.5 * (torch.exp(-10.0 * vz * vz) + torch.exp(-5.0 * wxy))


def base_height_exp(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, target: float = 0.98, gain: float = 100.0) -> torch.Tensor:
    """humanoid-gym base_height: exp(-gain |h - target|) on the root height."""
    asset = env.scene[asset_cfg.name]
    return torch.exp(-gain * (asset.data.root_pos_w[:, 2] - target).abs())


def feet_contact_forces_hinge(
    env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, max_force: float = 700.0, cap: float = 400.0
) -> torch.Tensor:
    """humanoid-gym feet_contact_forces: sum_f clip(||F_f|| - max_force, 0, cap). Menlo's 'place feet gently rather than stomping'."""
    _, f = _contact(env, sensor_cfg)
    return (f.norm(dim=-1) - max_force).clamp(min=0.0, max=cap).sum(dim=1)


def gait_clock(env: "ManagerBasedRLEnv", period: float = 0.8) -> torch.Tensor:
    """(sin, cos) of 2 pi phi, phi = episode time mod period -- the clock the phase-contact rewards run on, made VISIBLE to the policy
    (Unitree H1-2 walk 0.8 s, humanoid-gym 0.64 s, #83 0.8 s). Same formula as mdp.gait_phase; own name so the mirror map can key it."""
    phi = (env.episode_length_buf * env.step_dt) % period / period
    return torch.stack([torch.sin(2 * torch.pi * phi), torch.cos(2 * torch.pi * phi)], dim=1)
