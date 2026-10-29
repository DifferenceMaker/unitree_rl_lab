"""LM7H terms (2026-10-29): the HANDOVER and the ZERO-COMMAND STANCE.

Hardware 2026-10-29 (pc4, p14h_stancewide -> walk by the runtime switch): lm7f_kit_gforce_airtime "the initial handover made him start
stepping backwards and the policy wanted to keep moving backwards. Legs rotated inwards"; lm7f_kit_footbonus6_still "Stumbles at
handover and also at default stance". Operator: "Handover from balance to locomotion is EXTREMELY crucial. IT cannot step when handover
is there ... from arms being at any manipulation position"; "The standing at default stance (0 commanded velocity) has to be perfect. IT
has to stand like our balance policy."

Measured cause (the two env.yaml ledgers): p14h_stancewide stands with its feet 0.325 m apart (bank 2026-10-29: feet at y +0.159 /
-0.167, stagger < 1 cm) because its foot_stance_tracking nominal is +-0.15 and feet_too_near is 0.28; the walk parent's nominal is
+-0.10 (always on) with feet_too_near 0.22. At the switch the walk policy's own stance kernel pays exp(-0.12/0.08) = 0.22 of its 1.5
ceiling for the balance stance -> 1.17 / step for drawing the feet in = the inward step. Every term here is gated on the COMMAND (in the
observation; Bible rule 32): the walking ledger is untouched.

  foot_stance_tracking_split   the walk's stance kernel with the balance policy's nominal at a zero command, the walk's under a command
  feet_too_near_split / feet_too_far_split   the hinges at the balance policy's thresholds (.28 / .45) at a zero command
  feet_step_price_standing     p14h's feet_air_time_step_penalty (touchdown 0.4) at a zero command: a stop step stays affordable
  torso_still_lin / _ang_standing   per-axis stillness kernels sized to the parent's measured sway (Bible rule 31)
  reset_from_pose_bank_hold    the bank start + a ZERO COMMAND held for hold_s after it (the real handover: the switch only fires at a
                               zero command, the operator commands after it)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .lm7f import reset_from_pose_bank

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _standing(env, command_name: str, cmd_thr: float) -> torch.Tensor:
    return torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < cmd_thr


def _feet_b_xy(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    d = asset.data.body_pos_w[:, asset_cfg.body_ids, :3] - asset.data.root_pos_w[:, :3].unsqueeze(1)
    n = d.shape[1]
    q = asset.data.root_quat_w.unsqueeze(1).expand(-1, n, -1).reshape(-1, 4)
    return quat_apply_inverse(q, d.reshape(-1, 3)).reshape(-1, n, 3)[..., :2]


def foot_stance_tracking_split(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, std: float = 0.08,
                               nominal_stand: list = ((0.01, 0.16), (0.01, -0.16)), nominal_walk: list = ((0.0, 0.10), (0.0, -0.10)),
                               command_name: str = "base_velocity", cmd_thr: float = 0.03) -> torch.Tensor:
    """foot_stance_tracking (exp(-sum |foot_b - nominal| / std), linear exponent) with the nominal chosen by the command: the
    balance policy's measured stand at a zero command, the walk's own nominal under a command. Same ceiling (1.0) in both."""
    fb = _feet_b_xy(env, asset_cfg)
    ns = torch.tensor(nominal_stand, device=env.device, dtype=fb.dtype)
    nw = torch.tensor(nominal_walk, device=env.device, dtype=fb.dtype)
    st = _standing(env, command_name, cmd_thr).view(-1, 1, 1)
    nom = torch.where(st, ns.unsqueeze(0), nw.unsqueeze(0))
    return torch.exp(-torch.norm(fb - nom, dim=-1).sum(dim=-1) / std)


def _feet_dist(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    p = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    return torch.norm(p[:, 0] - p[:, 1], dim=-1)


def feet_too_near_split(env: "ManagerBasedRLEnv", threshold_stand: float = 0.28, threshold_walk: float = 0.22,
                        command_name: str = "base_velocity", cmd_thr: float = 0.03,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """feet_too_near (hinge (thr - planar foot distance)+) with the balance policy's 0.28 at a zero command. NEGATIVE weight."""
    thr = torch.where(_standing(env, command_name, cmd_thr), threshold_stand, threshold_walk)
    return (thr - _feet_dist(env, asset_cfg)).clamp(min=0.0)


def feet_too_far_split(env: "ManagerBasedRLEnv", threshold_stand: float = 0.45, threshold_walk: float = 0.6,
                       command_name: str = "base_velocity", cmd_thr: float = 0.03,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """feet_too_far (hinge (planar foot distance - thr)+) with the balance policy's 0.45 at a zero command (operator 10-29 on
    lm7g_energy_gated: "Legs drift apart. The term for legs drifting apart needs to be increased"). NEGATIVE weight."""
    thr = torch.where(_standing(env, command_name, cmd_thr), threshold_stand, threshold_walk)
    return (_feet_dist(env, asset_cfg) - thr).clamp(min=0.0)


def feet_step_price_standing(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, touchdown_penalty: float = 0.4,
                             command_name: str = "base_velocity", cmd_thr: float = 0.03) -> torch.Tensor:
    """p14h_stancewide's feet_air_time_step_penalty (each touchdown costs `touchdown_penalty`, standing still costs 0) charged only
    at a zero command. NEGATIVE weight. At -1: a 2-step stop costs 0.8 once, against alive 30 / step -- a stop stays affordable, a
    stepping stand does not stay free."""
    cs = env.scene.sensors[sensor_cfg.name]
    ct = cs.data.current_contact_time[:, sensor_cfg.body_ids]
    landed = (ct > 0.0) & (ct <= env.step_dt + 1e-6)
    return (landed.float() * touchdown_penalty).sum(dim=1) * _standing(env, command_name, cmd_thr).float()


def torso_still_lin_standing(env: "ManagerBasedRLEnv", std: float = 0.04, command_name: str = "base_velocity", cmd_thr: float = 0.03,
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"])) -> torch.Tensor:
    """exp(-|v_xy torso|^2 / std^2) at a zero command: the side-to-side / fore-aft SWAY as a kernel (the parent's L2 torso_lin_vel_xy
    -3 is flat at a sway speed of a few cm/s: 3 x 0.03^2 = 0.003 / step). POSITIVE weight."""
    asset = env.scene[asset_cfg.name]
    v = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :2]
    return torch.exp(-torch.sum(v * v, dim=-1) / std**2) * _standing(env, command_name, cmd_thr).float()


def torso_still_ang_standing(env: "ManagerBasedRLEnv", std: float = 0.09, command_name: str = "base_velocity", cmd_thr: float = 0.03,
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"])) -> torch.Tensor:
    """exp(-|w torso|^2 / std^2) at a zero command (all three axes: roll/pitch rocking and the yaw wobble). POSITIVE weight."""
    asset = env.scene[asset_cfg.name]
    w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :]
    return torch.exp(-torch.sum(w * w, dim=-1) / std**2) * _standing(env, command_name, cmd_thr).float()


def reset_from_pose_bank_hold(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, bank: str, prob: float = 0.5, pos_noise: float = 0.03,
                              vel_noise: float = 0.3, hold_s: float = 3.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """reset_from_pose_bank + the handover's ZERO COMMAND: the picked envs get `hold_s` seconds of a zero velocity command (applied by
    handover_command_hold every step, after the command manager's own resample), then a fresh command. Other envs: the trunk's reset."""
    pick = reset_from_pose_bank(env, env_ids, bank=bank, prob=prob, pos_noise=pos_noise, vel_noise=vel_noise, asset_cfg=asset_cfg)
    if not hasattr(env, "_lm7h_hold"):
        env._lm7h_hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._lm7h_bank_starts = 0
    env._lm7h_hold[env_ids] = 0
    if pick is not None and pick.numel() > 0:
        env._lm7h_hold[pick] = int(round(hold_s / env.step_dt))
        env._lm7h_bank_starts += int(pick.numel())


def handover_command_hold(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, command_name: str = "base_velocity"):
    """interval event (every step, after the command manager's compute): zero the velocity command of the envs in a handover hold and
    count the hold down; at its end the term resamples (time_left = 0), so the policy then gets an ordinary command."""
    h = getattr(env, "_lm7h_hold", None)
    if h is None:
        return
    term = env.command_manager.get_term(command_name)
    on = h > 0
    if env.common_step_counter % 2400 == 1:   # proof of life in the training log (JOB_VALIDITY 13: an event has no reward income)
        held = int(on.sum()); z = float(term.vel_command_b[on].abs().max()) if held else 0.0
        print(f"[lm7h] handover hold: {held} envs held now (max |cmd| {z:.3f}), bank starts so far {getattr(env, '_lm7h_bank_starts', 0)}", flush=True)
    if on.any():
        term.vel_command_b[on] = 0.0
        if hasattr(term, "is_standing_env"):
            term.is_standing_env[on] = True
        h[on] -= 1
        ended = on & (h == 0)
        if ended.any():
            term.time_left[ended] = 0.0
