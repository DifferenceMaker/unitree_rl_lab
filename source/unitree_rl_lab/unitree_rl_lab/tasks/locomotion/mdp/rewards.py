from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def feet_air_time_step_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    touchdown_penalty: float = 0.4,
    free_when_displaced_m: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Digit-style stepping regularizer (Oregon State, arXiv 2404.19173).
    free_when_displaced_m (dp8, 2026-09-09): touchdowns are priced only while
    the base is within this radius of its anchor (env.spawn_root_xy); farther
    out the step toward the anchor is FREE — the transit must not be taxed by
    the stillness regularizer that made the shimmy the cheaper move. None =
    legacy, always priced.

    Standing perfectly still => feet never lift => no touchdown => returns 0
    (stillness is free). Each foot touchdown costs `touchdown_penalty`,
    regardless of stride length. Standing still costs 0; any step costs a fixed
    amount. Use with a NEGATIVE weight. alive reward dominates a needed step.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    just_landed = (current_contact_time > 0.0) & (current_contact_time <= env.step_dt + 1e-6)
    # Count touchdowns; standing still => no touchdown => 0. NEGATIVE weight =>
    # each step costs a fixed amount regardless of stride length (want NO
    # stepping, not long strides). alive=30 dominates a needed recovery step.
    touchdowns = just_landed.float() * touchdown_penalty
    pen = torch.sum(touchdowns, dim=1)
    if free_when_displaced_m is not None and hasattr(env, "spawn_root_xy"):
        asset = env.scene[asset_cfg.name]
        near = torch.norm(asset.data.root_pos_w[:, :2] - env.spawn_root_xy, dim=-1) < free_when_displaced_m
        pen = pen * near.float()
    return pen


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward

def stance_bonus(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg",
    std: float = 0.15,
) -> "torch.Tensor":
    """Positive bonus for joint positions near default pose.

    Returns exp(-||pos - default||^2 / std^2) per env. Bonus ~1.0 at default,
    falls off rapidly with deviation. Use POSITIVE weight in RewardsCfg.
    Distinct from joint_deviation_l1 (penalty) — this creates an active
    gradient pulling the policy toward default pose.
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or (isinstance(joint_ids, slice) and joint_ids == slice(None)):
        current = asset.data.joint_pos
        default = asset.data.default_joint_pos
    else:
        current = asset.data.joint_pos[:, joint_ids]
        default = asset.data.default_joint_pos[:, joint_ids]
    deviation = current - default
    sq_dist = torch.sum(deviation * deviation, dim=-1)
    return torch.exp(-sq_dist / (std * std))# ============================================================
# ADDITIONS to rewards.py for arm curriculum (option I).
# Append these to the existing rewards.py — do NOT replace the file.
# ============================================================


def arm_target_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: "SceneEntityCfg",
    std: float = 0.20,
) -> "torch.Tensor":
    """Reward tracking the commanded arm joint pose.

    Returns exp(-||target - actual||^2 / std^2) per env, in [0, 1].
    Peaks at 1.0 when the arm joint positions match the command exactly,
    falls off as deviation grows. Use POSITIVE weight in RewardsCfg.

    The std parameter controls how forgiving the reward is. std=0.20 rad
    means the reward is ~0.78 when each joint is off by 0.10 rad on
    average, and ~0.37 when each is off by 0.20 rad. With 14 joints,
    the squared-distance scales accordingly so std should be set
    relative to expected per-joint error.
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or (isinstance(joint_ids, slice) and joint_ids == slice(None)):
        actual = asset.data.joint_pos
    else:
        actual = asset.data.joint_pos[:, joint_ids]

    # The command is the absolute target arm pose
    target = env.command_manager.get_command(command_name)

    deviation = actual - target
    sq_dist = torch.sum(deviation * deviation, dim=-1)
    return torch.exp(-sq_dist / (std * std))


def body_lin_vel_xy_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Penalize horizontal linear velocity of a specific body (not the base).

    Used for torso (head proxy) stability. Returns sum of squared lin vel
    components in world XY plane. Use NEGATIVE weight in RewardsCfg.

    asset_cfg.body_ids must resolve to exactly one body — we use [:, body_id, :2]
    to get xy components.
    """
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    # body_lin_vel_w shape: (num_envs, num_bodies, 3)
    lin_vel_xy = asset.data.body_lin_vel_w[:, body_ids, :2]
    # Sum over bodies (usually just one) and over xy components
    return torch.sum(torch.sum(lin_vel_xy * lin_vel_xy, dim=-1), dim=-1)


def body_ang_vel_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Penalize angular velocity of a specific body.

    Used for torso (head proxy) stability — keep camera from rotating.
    Returns sum of squared angular velocity over all axes. Use NEGATIVE
    weight in RewardsCfg.
    """
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    # body_ang_vel_w shape: (num_envs, num_bodies, 3)
    ang_vel = asset.data.body_ang_vel_w[:, body_ids, :]
    return torch.sum(torch.sum(ang_vel * ang_vel, dim=-1), dim=-1)


def action_rate_l2_scoped(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg" = None,
) -> "torch.Tensor":
    """Penalize action change rate, scoped to specific joint indices.

    Differs from isaaclab.envs.mdp.action_rate_l2 by accepting an
    asset_cfg whose resolved joint_ids map to specific action dimensions.
    Used to penalize smoothness on legs+torso only, leaving arms free
    to make large motions for tracking.
    """
    if asset_cfg is None:
        from isaaclab.managers import SceneEntityCfg
        asset_cfg = SceneEntityCfg("robot")

    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        return torch.sum(
            torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
        )

    action = env.action_manager.action[:, joint_ids]
    prev_action = env.action_manager.prev_action[:, joint_ids]
    return torch.sum(torch.square(action - prev_action), dim=1)
"""Torso stability bonus reward — append to mdp/rewards.py.

Positive bonus for keeping the torso (and thus the head/camera) still.
Falls off exponentially with combined linear and angular velocity.

Distinct from torso_lin_vel_l2 / torso_ang_vel_l2 which are pure penalties
on motion magnitude. This term creates a positive gradient toward "be still"
rather than just a cost gradient against motion.

Similar to stance_bonus_legs_torso: at zero motion the reward is +1.0,
falls off rapidly with motion. Weight +1.5 means stillness contributes
0.75-1.5 to per-step reward, comparable to other shaping terms.
"""

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def torso_stability_bonus(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    std_lin: float = 0.15,
    std_ang: float = 0.30,
    command_name: str | None = None,
) -> torch.Tensor:
    """Bonus reward for stable torso (head/camera).

    Args:
        asset_cfg: SceneEntityCfg selecting torso body.
        std_lin: Std for linear velocity falloff (m/s). At 0.15, bonus is
                 ~e^(-1) = 0.37 when |lin_vel|=0.15 m/s.
        std_ang: Std for angular velocity falloff (rad/s). At 0.30, bonus is
                 ~e^(-1) = 0.37 when |ang_vel|=0.30 rad/s.

    Returns:
        Per-env bonus tensor in [0, 1].
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get torso body velocities (world frame)
    body_ids = asset_cfg.body_ids
    lin_vel = asset.data.body_lin_vel_w[:, body_ids, :2]  # XY only — vertical motion is gravity
    ang_vel = asset.data.body_ang_vel_w[:, body_ids, :]   # All 3 axes
    if command_name is not None:
        # lm5e_turn (2026-08-28 yaw audit): the ungated 3-axis kernel forfeits the
        # WHOLE +6 at a commanded 0.5 rad/s turn (exp(-(0.5/0.15)^2) ~ 0) — the
        # single largest tax on rotating. Score yaw RELATIVE to the commanded wz
        # (upright body: world z ~ body z); roll/pitch stay absolute.
        wz_cmd = env.command_manager.get_command(command_name)[:, 2]
        ang_vel = ang_vel.clone()
        ang_vel[:, :, 2] = ang_vel[:, :, 2] - wz_cmd.unsqueeze(1)

    # Squared norms per env
    lin_sq = torch.sum(lin_vel ** 2, dim=-1).squeeze(-1)  # (num_envs,)
    ang_sq = torch.sum(ang_vel ** 2, dim=-1).squeeze(-1)  # (num_envs,)

    # Combined exponential bonus
    bonus = torch.exp(-(lin_sq / (std_lin ** 2) + ang_sq / (std_ang ** 2)))
    return bonus
def foot_stance_tracking(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg",
    std: float = 0.08,
    nominal_foot_pos_b: list[list[float]] = [[0.0, 0.10], [0.0, -0.10]],
) -> "torch.Tensor":
    """Positive bonus for keeping feet near nominal stance positions.

    Returns exp(-||displacement||/std) per env, summed across feet.
    At nominal stance bonus ≈ 1.0 × weight. At 8cm displacement drops
    to 0.37; at 16cm to 0.14; at 30cm to 0.024. Creates "small steps
    cheap, big steps expensive" gradient.

    Args:
        asset_cfg: must specify foot body_names, e.g.
            SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]).
        std: stance tracking precision. 0.08 m default.
        nominal_foot_pos_b: list of (x, y) per foot in body frame at
            the default standing pose. Order MUST match asset_cfg.body_ids
            order. Default values are H1-2 approximate, verify by
            dumping `body_pos_w - root_pos_w` at env reset.

    Use as positive bonus reward; suggested weight +1.0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    nominal = torch.tensor(nominal_foot_pos_b, device=env.device, dtype=torch.float32)

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    root_pos_w = asset.data.root_pos_w[:, :3].unsqueeze(1)
    delta_w = foot_pos_w - root_pos_w

    # Transform world delta to body frame (xy only)
    foot_pos_b_xy = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 2, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        body_frame_3d = quat_apply_inverse(asset.data.root_quat_w, delta_w[:, i, :])
        foot_pos_b_xy[:, i] = body_frame_3d[:, :2]

    displacement = torch.norm(foot_pos_b_xy - nominal.unsqueeze(0), dim=-1)
    return torch.exp(-displacement.sum(dim=-1) / std)

def heading_l2_from_spawn(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared deviation of current yaw from spawn yaw.

    Wraps yaw delta to [-pi, pi] to handle the circular nature of yaw.
    Returned per-env reward magnitude is in radians^2; with weight=-2.0, a
    sustained 0.3 rad (~17°) yaw drift gives reward of -0.18 per step.
    """
    if not hasattr(env, "spawn_yaw"):
        # Before first reset (shouldn't happen in normal flow)
        return torch.zeros(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    current_yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )

    # Wrap delta to [-pi, pi]
    delta = current_yaw - env.spawn_yaw
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))

    return delta ** 2


def base_pos_xy_l2_from_spawn(
    env: "ManagerBasedRLEnv",
    command_name: str | None = None,
    lean_height: float = 0.87,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared L2 distance of base xy from spawn xy.

    Magnitudes scale with displacement squared: 0.5m drift = 0.25 reward unit
    (with weight=-1.0, penalty = -0.25 per step).
    """
    if not hasattr(env, "spawn_root_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    current_xy = asset.data.root_pos_w[:, :2]
    delta = current_xy - env.spawn_root_xy
    d_lean, fwd_lean = _lean_fwd_shift(env, command_name, lean_height)
    if d_lean is not None:
        delta = delta - d_lean.unsqueeze(-1) * fwd_lean
    return torch.sum(delta ** 2, dim=-1)


def base_forward_zone_penalty(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.10,
    command_name: str | None = None,
    lean_height: float = 0.87,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DESK LINE: penalize forward base drift into the desk zone (asymmetric).

    Forward = the spawn heading (+x in the spawn frame). Only forward drift
    BEYOND `threshold` is penalized (one-sided linear hinge) — the robot may
    sway / step laterally or backward freely, but drifting toward the desk in
    front costs. This is the "geographic desk zone" penalty: dense (a gradient
    that eases the base away), NOT a termination (a shove into the desk is not
    a failure to end the episode on). Pairs with the light symmetric
    base_pos_xy_l2_from_spawn. Needs env.spawn_root_xy + env.spawn_yaw
    (capture_spawn_state).
    """
    if not hasattr(env, "spawn_root_xy") or not hasattr(env, "spawn_yaw"):
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    delta = asset.data.root_pos_w[:, :2] - env.spawn_root_xy
    fwd = torch.stack([torch.cos(env.spawn_yaw), torch.sin(env.spawn_yaw)], dim=-1)
    forward_disp = torch.sum(delta * fwd, dim=-1)
    d_lean, _ = _lean_fwd_shift(env, command_name, lean_height)
    if d_lean is not None:
        # dp6c: the commanded lean legitimately carries the base d(theta) forward
        forward_disp = forward_disp - d_lean
    return torch.clamp(forward_disp - threshold, min=0.0)


def foot_displacement_l2_from_spawn(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
) -> torch.Tensor:
    """Penalize squared L2 displacement of feet from spawn foot positions.

    Sums over both feet and xy. A foot stepping 0.3m gives reward unit 0.09
    (with weight=-0.5, penalty = -0.045 per step). Multiple steps compound.
    """
    if not hasattr(env, "spawn_foot_pos") or asset_cfg.body_ids is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    current_foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    delta = current_foot_pos - env.spawn_foot_pos
    # Sum over feet and xy
    return torch.sum(delta ** 2, dim=(-2, -1))

def heading_stable_bonus(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positive exponential reward for heading near spawn heading.
    
    Companion to heading_l2_from_spawn penalty. Reward shape:
        exp(-|yaw_drift|² / std²)
    
    With std=0.1: 0.1 rad (~6°) drift gives ~0.37 reward; 0.3 rad → ~0.011.
    """
    if not hasattr(env, "spawn_yaw"):
        return torch.zeros(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    current_yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
    )
    delta = current_yaw - env.spawn_yaw
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    return torch.exp(-(delta ** 2) / (std ** 2))


def centered_pose_bonus(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positive exponential reward for base xy near spawn xy.
    
    Companion to base_pos_xy_l2_from_spawn penalty. With std=0.15:
    0.15m drift → ~0.37 reward; 0.5m drift → ~0.0001.
    """
    if not hasattr(env, "spawn_root_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    current_xy = asset.data.root_pos_w[:, :2]
    delta_sq = torch.sum((current_xy - env.spawn_root_xy) ** 2, dim=-1)
    return torch.exp(-delta_sq / (std ** 2))


def foot_planted_bonus(
    env: "ManagerBasedRLEnv",
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
) -> torch.Tensor:
    """Positive exponential reward for feet near spawn positions.
    
    Companion (or replacement) for foot_displacement_l2_from_spawn. With std=0.05:
    tight tolerance — 5cm displacement → ~0.37 reward; 10cm → ~0.018.
    """
    if not hasattr(env, "spawn_foot_pos") or asset_cfg.body_ids is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    current_foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    delta_sq = torch.sum((current_foot_pos - env.spawn_foot_pos) ** 2, dim=(-2, -1))
    return torch.exp(-delta_sq / (std ** 2))


def upright_bonus(
    env: "ManagerBasedRLEnv",
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Positive exponential reward for torso staying vertical.
    
    Companion to flat_orientation_l2 penalty. Measures projected gravity in
    base frame — when robot is upright, projected_gravity_b is [0, 0, -1] so
    xy magnitude is ~0. With std=0.05: ~3° tilt → ~0.37 reward.
    """
    asset = env.scene[asset_cfg.name]
    proj_gravity_xy_sq = torch.sum(asset.data.projected_gravity_b[:, :2] ** 2, dim=-1)
    return torch.exp(-proj_gravity_xy_sq / (std ** 2))


def upright_linear(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """LINEAR uprightness (dp4c_armdesk_linear, operator 2026-08-06): reward
    proportional to how upright the base is — cos(tilt) = -g_z in the base
    frame, clamped to [0,1]. Unlike upright_bonus's kernel (desk std 0.01 =
    total forfeiture beyond ~1 deg), there is no cliff: a 20-deg working lean
    keeps ~94% of the reward — "some uprightness still pays" while leaning
    costs progressively, never catastrophically. Bounded [0,1]."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0)


def flat_orientation_linear(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """LINEAR anti-lean penalty: |g_xy| = sin(tilt), in [0,1]. The L2 version
    is quadratic — negligible cost near upright, a WALL at working-lean depth;
    this one prices each degree of lean about the same, so a deliberate desk
    lean is affordable while exactly-upright stays cheapest. Use INSTEAD of
    flat_orientation_l2 (weight it negative). Bounded [0,1]."""
    asset = env.scene[asset_cfg.name]
    return torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1)

def foot_impact_velocity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """p12 soft-landing shaper: downward foot speed at the touchdown step.

    Hardware motivation: hard heel strikes on cement peeled the foot rubber.
    feet_air_time_step_penalty makes steps RARE; this makes the remaining
    steps SOFT — the policy learns to decelerate the foot before contact.
    Standing still => no touchdowns => 0 (stillness stays free). Use with a
    NEGATIVE weight. Same just-landed bookkeeping as feet_air_time_step.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    asset = env.scene[asset_cfg.name]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    just_landed = (current_contact_time > 0.0) & (current_contact_time <= env.step_dt + 1e-6)
    # Downward (negative z) world-frame velocity of the foot bodies, clamped to
    # descent only — upward motion at grazing contact is not an impact.
    vz = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    impact_speed = torch.clamp(-vz, min=0.0) * just_landed.float()
    return torch.sum(impact_speed, dim=1)


def foot_impact_velocity_peak(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    max_speed: float = 3.0,
) -> torch.Tensor:
    """softland v2 (code review 2026-09-03): foot_impact_velocity reads the
    foot's velocity AT the policy step where the touchdown is first visible —
    up to 4 physics substeps after contact, when the ground has already
    arrested the foot, so it returned ~0 on almost every landing ("softland
    indistinguishable in sim", retired 07-14 as silent).

    This version LATCHES the peak downward speed of each foot DURING THE SWING
    (per policy step, max over the flight) and charges that latched value once
    at touchdown. Standing still => no touchdowns => 0 (stillness stays free).
    Clamped at max_speed (Bible: bounded penalties). NEGATIVE weight.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0   # (N, F)
    vz_down = torch.clamp(-asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2], min=0.0)  # (N, F)
    if not hasattr(env, "_fivp_peak"):
        env._fivp_peak = torch.zeros_like(vz_down)
        env._fivp_prev_contact = torch.ones_like(in_contact)
    if hasattr(env, "reset_buf") and env.reset_buf.any():
        rb = env.reset_buf.bool().unsqueeze(-1)
        env._fivp_peak = torch.where(rb, torch.zeros_like(env._fivp_peak), env._fivp_peak)
        env._fivp_prev_contact = torch.where(rb, torch.ones_like(in_contact), env._fivp_prev_contact)
    airborne = ~in_contact
    env._fivp_peak = torch.where(airborne, torch.maximum(env._fivp_peak, vz_down), env._fivp_peak)
    touchdown = ~env._fivp_prev_contact & in_contact
    charged = torch.clamp(env._fivp_peak, max=max_speed) * touchdown.float()
    env._fivp_peak = torch.where(touchdown, torch.zeros_like(env._fivp_peak), env._fivp_peak)
    env._fivp_prev_contact = in_contact.clone()
    return torch.sum(charged, dim=1)


def foot_contact_force_hinge(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold_n: float = 950.0,
    max_excess_n: float = 1000.0,
) -> torch.Tensor:
    """Calibrated impact-force hinge (code review 2026-09-03): zero below
    threshold_n, linear on the excess, CLAMPED at max_excess_n (the IsaacLab
    hinge pattern was unbounded in newtons on a signal that spikes to kN).

    THE CALIBRATION LAW: threshold must sit ABOVE static single-support load
    (77 kg => ~755 N — the whole robot on one foot, zero impact), else the
    term charges for the existence of a stance phase and buys shuffling
    (contact500 hardware verdict 08-06: "it slides around"). Default 950 N
    ~= 1.25x single support (the Humanoid-Gym ratio). Reads the MAX over the
    sensor history buffer — requires history_length == decimation (4), else
    intra-step peaks fall out of the buffer unread. NEGATIVE weight.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    hist = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]   # (N, T, F, 3)
    peak = hist.norm(dim=-1).amax(dim=1)                                    # (N, F) max over history
    excess = torch.clamp(peak - threshold_n, min=0.0, max=max_excess_n)
    return torch.sum(excess, dim=1)


def feet_flat_orientation(
    env: ManagerBasedRLEnv,
    height_thresh: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize tilted feet near the ground (HuB feet-orientation term).

    For each foot body below height_thresh, the penalty is the xy-norm of the
    gravity direction expressed in the foot frame (0 = sole level, grows with
    tilt). Targets edge-first / heel-first strikes: a tilted touchdown slaps
    (loudness) and shrinks the support patch regardless of descent speed --
    the failure mode the impact-VELOCITY proxy (softland) missed.
    """
    from isaaclab.utils.math import quat_apply_inverse

    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    n_feet = quat.shape[1]
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, n_feet, 3)
    g_foot = quat_apply_inverse(quat.reshape(-1, 4), g_world.reshape(-1, 3)).reshape(env.num_envs, n_feet, 3)
    tilt = torch.norm(g_foot[..., :2], dim=-1)
    near = (pos_z < height_thresh).float()
    return (tilt * near).sum(dim=1)


def capture_point_touchdown_distance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    max_dist: float = 0.5,
    com_offset_b: tuple = (0.0, 0.0),
) -> torch.Tensor:
    """p12/p13 step-placement shaper: distance from the landing foot to the
    instantaneous capture point, charged once per touchdown.

    Capture point: cp = com_xy + v_xy * sqrt(h/g) — the spot where planting
    the foot absorbs the current momentum in ONE step. Landing far from cp is
    the overshoot/multi-step signature (the "Figure robots don't overshoot"
    observation). PENALTY form on purpose: a placement BONUS per touchdown
    would let the policy farm reward by stepping in place onto cp; a penalty
    can never make stepping profitable — stillness = 0, a perfectly placed
    needed step ~ 0, an overshot step pays. Use with a NEGATIVE weight.
    Root link is the CoM proxy (standing humanoid). Distance clamped so one
    wild step can't dominate the return.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    asset = env.scene[asset_cfg.name]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    just_landed = (current_contact_time > 0.0) & (current_contact_time <= env.step_dt + 1e-6)

    root_xy = asset.data.root_pos_w[:, :2]
    # CoM-proxy correction (2026-07-12): the pelvis root sits ~2 cm BEHIND the
    # true whole-body CoM (measured +0.0198 m at default pose) — a pelvis-based
    # capture point teaches aft-biased foot placement ("recovers, then falls
    # backward"). com_offset_b shifts the proxy in the BASE frame (x forward),
    # rotated into world by base yaw. (0,0) preserves the original behavior.
    if com_offset_b[0] != 0.0 or com_offset_b[1] != 0.0:
        from isaaclab.utils.math import quat_apply_yaw
        offset3 = torch.zeros(env.num_envs, 3, device=env.device)
        offset3[:, 0] = float(com_offset_b[0])
        offset3[:, 1] = float(com_offset_b[1])
        root_xy = root_xy + quat_apply_yaw(asset.data.root_quat_w, offset3)[:, :2]
    v_xy = asset.data.root_lin_vel_w[:, :2]
    h = torch.clamp(asset.data.root_pos_w[:, 2], min=0.3)
    cp_xy = root_xy + v_xy * torch.sqrt(h / 9.81).unsqueeze(-1)

    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    dist = torch.norm(foot_xy - cp_xy.unsqueeze(1), dim=-1).clamp(max=max_dist)
    return torch.sum(dist * just_landed.float(), dim=1)


def foot_homing_huber(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
    delta: float = 0.15,
) -> torch.Tensor:
    """p12e homing-v2: Huber-shaped pull of the feet toward their spawn spots.

    The p12d homing L2 boost detonated when the 50 N sustained-push curriculum
    forced large displacements (quadratic penalty cliff -> PPO collapse).
    Huber keeps the L2's precision near home but saturates the GRADIENT:
        f(d) = d^2 / (2*delta)      for d <= delta   (quadratic, slope d/delta)
        f(d) = d - delta/2          for d >  delta   (linear, slope 1)
    Gradient everywhere ("breadcrumbs"), bounded cost anywhere (no cliff under
    forced displacement). Per foot, xy distance to spawn; summed. Use with a
    NEGATIVE weight.
    """
    if not hasattr(env, "spawn_foot_pos") or asset_cfg.body_ids is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    current_foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    d = torch.norm(current_foot_pos - env.spawn_foot_pos, dim=-1)   # (N, feet)
    huber = torch.where(d <= delta, d * d / (2.0 * delta), d - delta / 2.0)
    return torch.sum(huber, dim=-1)


def anchor_hold_l2(
    env: "ManagerBasedRLEnv",
    fwd_offset: float = 0.5,
    heading_scale: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DESK LINE v2: strong symmetric attractor to the start pose, anchor-framed.

    Squared xy distance from spawn PLUS squared heading error toward the anchor
    (fwd_offset ahead of spawn — 'keep the camera on the point'). Quadratic =>
    near-zero gradient at zero error (free micro-sway) and a steep
    return-to-start-pose pull when displaced. Weight it to DOMINATE the zone /
    homing terms: it converts 'anywhere behind the wall is fine' (the fz6
    backlean, 2026-07-24) into 'exactly HERE is the sweet spot'.
    """
    if not hasattr(env, "spawn_root_xy") or not hasattr(env, "spawn_yaw"):
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    delta = asset.data.root_pos_w[:, :2] - env.spawn_root_xy
    pos_err2 = torch.sum(delta * delta, dim=-1)
    fwd = torch.stack([torch.cos(env.spawn_yaw), torch.sin(env.spawn_yaw)], dim=-1)
    to_anchor = env.spawn_root_xy + fwd_offset * fwd - asset.data.root_pos_w[:, :2]
    desired_yaw = torch.atan2(to_anchor[:, 1], to_anchor[:, 0])
    q = asset.data.root_quat_w
    yaw = torch.atan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]),
    )
    yaw_err = torch.atan2(torch.sin(desired_yaw - yaw), torch.cos(desired_yaw - yaw))
    return pos_err2 + heading_scale * yaw_err * yaw_err


def base_radial_zone_penalty(
    env: "ManagerBasedRLEnv",
    threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DESK LINE: radial stay-put hinge — free inside `threshold` in ANY
    direction, linear cost outside. The all-around companion to the one-sided
    base_forward_zone_penalty (which stays as the harder desk wall on top);
    closes the free-backward-drift gap (sim2sim 2026-07-24). Same dense
    not-a-termination philosophy.
    """
    if not hasattr(env, "spawn_root_xy"):
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    d = torch.norm(asset.data.root_pos_w[:, :2] - env.spawn_root_xy, dim=-1)
    return torch.clamp(d - threshold, min=0.0)


def planted_in_zone_bonus(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """DESK LINE 'boxing' bonus: +1 per control tick while BOTH feet are in
    contact AND the base is inside the radial zone. Pays for absorbing pushes
    planted; stops paying the tick a foot lifts or ground is conceded. NOT a
    stepping penalty — the fall-avoidance step stays available, it just earns
    nothing while airborne/displaced (no slide loophole: sliding out of the
    zone also stops the pay).
    """
    if not hasattr(env, "spawn_root_xy"):
        return torch.zeros(env.num_envs, device=env.device)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    both_planted = (forces.norm(dim=-1) > 1.0).all(dim=-1)
    asset = env.scene[asset_cfg.name]
    d = torch.norm(asset.data.root_pos_w[:, :2] - env.spawn_root_xy, dim=-1)
    return (both_planted & (d < threshold)).float()


def action_magnitude_over(
    env: "ManagerBasedRLEnv",
    threshold: float = 3.0,
    max_excess: float = 15.0,
    square: bool = False,
    mode: str = "hinge",
    tau: float = 1.0,
    power: float = 3.0,
) -> torch.Tensor:
    """Hinged penalty on RAW action MAGNITUDE: zero below `threshold`, linear
    (or squared) on the excess, clamped. Sum over action dims.

    THE GAP IT FILLS (measured 2026-09-07 on p13d_armknees): the balance ledger
    prices action CHANGE (`action_rate`), joint acceleration, and the resulting
    joint POSITION (`dof_pos_limits`) — but NOTHING prices how big an action is.
    A policy may hold |a| = 17 forever at zero action_rate cost. Measured
    distribution: p50 0.86, p90 3.49, p99 11.58, max 17.37, and 2.1% of samples
    above 10. With `scale` 0.25 an action of 17 asks for 4.3 rad off the default
    joint position.

    WHY IT MATTERS MORE THAN IT LOOKS: the worst dims are the KNEES, the only
    joints carrying an action clip (target clamped to [0.45, 2.5] rad). The clip
    absorbs the excess, so the huge action costs nothing AND the clipped target
    never violates `dof_pos_limits` (-20) — the clip SHIELDS the policy from the
    penalty that would otherwise have caught it. Clipping treats the symptom;
    this term prices the cause. Use the two together (raise/remove the clip and
    price the magnitude), or the policy simply keeps slamming the limit.

    threshold: what counts as a "normal" action. At 1.0 roughly HALF of today's
    samples pay (a cost, and a large behavioural reshape — expect a transient and
    budget more iterations); at 3.0 about 12% pay; at 10.0 only the ~2%
    pathological tail pays (a true constraint, nearly free otherwise).
    SHAPES (operator 2026-09-07: "making the disincentivising after a threshold
    doesn't disincentivise taking smaller actions"). The hinge has EXACTLY ZERO
    gradient below `threshold`, so a policy sitting at |a| = 2.9 feels nothing
    and never moves toward 1.0 — only the excess is priced. The other two modes
    keep a nonzero (but small) gradient everywhere:

      mode="hinge"     relu(|a| - thr)              grad 0 below thr, 1 above
      mode="softplus"  tau*log(1 + e^((|a|-thr)/tau))   grad sigmoid((|a|-thr)/tau):
                       0.12 at |a|=1, 0.5 AT the threshold, ->1 above. Gentle
                       pressure everywhere, ramps hard at thr, bounded gradient.
                       This is the operator's "doesn't penalise at low, ramps up
                       quick at 3.0" — and it IS a logarithm (softplus), unlike a
                       plain log(), which is CONCAVE and would push hardest on the
                       SMALL actions and barely notice |a|=17.
      mode="power"     (|a| / thr)^power            grad grows without bound:
                       33x (p=3) at |a|=17 vs 1.0 at the threshold. The tail-killer.

    threshold: what counts as a "normal" action. At 1.0 roughly HALF of today's
    samples pay (a cost, and a large behavioural reshape — expect a transient and
    budget more iterations); at 3.0 about 12% pay; at 10.0 only the ~2%
    pathological tail pays.
    square: legacy hinge variant (excess^2). NEGATIVE weight for every mode.
    """
    a = env.action_manager.action.abs()
    if mode == "softplus":
        pen = tau * torch.nn.functional.softplus((a - threshold) / tau)
        return pen.clamp(max=max_excess).sum(dim=-1)
    if mode == "power":
        pen = (a / threshold).pow(power)
        return pen.clamp(max=max_excess * max_excess).sum(dim=-1)
    exc = torch.clamp(a - threshold, min=0.0, max=max_excess)
    if square:
        exc = exc.square()
    return exc.sum(dim=-1)


def joint_torque_over_limit(
    env: "ManagerBasedRLEnv",
    limit_nm: float = 80.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hinged over-torque penalty: zero below `limit_nm`, linear on the excess.

    The burnout guard (hardware 2026-07-25): sym_sharp locked into an isometric
    hip-roll squeeze at 102/136 Nm for ~4.5 min, heating 52->75C (~5C/min) until
    the Unitree safety damped the leg. The FUNCTIONAL lateral band (~35-70 Nm,
    thermally flat) must stay free — blanket tau^2 (latloop) punished it and the
    policy could not stand. This hinge prices ONLY the pathological band: apply
    to hip_roll (and optionally ankle_roll) via asset_cfg joint_names.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.clamp(tau.abs() - limit_nm, min=0.0).sum(dim=-1)


def _lean_fwd_shift(env, command_name: str | None, lean_height: float):
    """dp6c (2026-08-31): the EXPECTED base xy shift under the commanded lean.
    A pelvis pitched theta rad about the ankles moves the base ~lean_height*sin(theta)
    along the heading. The desk position terms (anchor_hold, base_forward_zone,
    base_pos_xy_from_spawn, base_height) scored that unavoidable displacement as
    drift: at 0.35 rad the anchor kernel alone forfeited ~12/step against the lean
    tracker's +1 — leaning could never pay (dp6b_leancmd verdict: "the leaning
    appears but it is really poorly tracked"). Shifting each term's REFERENCE by
    d(theta) makes lean and anchor-hold simultaneously satisfiable at FULL
    strength: the feet stay planted, only the body pivot is licensed, and at
    cmd=0 every term is bit-identical to the trunk (which also removes the
    learned standing lean at zero). Returns (d, fwd_unit) or (None, None)."""
    if command_name is None:
        return None, None
    theta = env.command_manager.get_command(command_name)[:, 0]
    d = lean_height * torch.sin(theta)
    fwd = torch.stack([torch.cos(env.spawn_yaw), torch.sin(env.spawn_yaw)], dim=-1)
    return d, fwd


def anchor_hold_bonus(
    env: "ManagerBasedRLEnv",
    fwd_offset: float = 0.5,
    heading_scale: float = 0.5,
    sigma: float = 0.15,
    heading_to: str = "bearing",
    command_name: str | None = None,
    lean_height: float = 0.87,
    deadband: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """ANCHOR ROUND 2: bounded attractor kernel, exp(-(pos_err^2 + hs*yaw_err^2)/sigma^2).
    deadband (dp8, operator 2026-09-09: 'the policy doesn't stop until it
    reaches the minimum allowed error... 5cm? 10cm?'): positional tolerance in
    metres — the kernel is FLAT (full income, zero gradient) inside it and the
    pull starts at its edge. Kills the shimmy-to-zero: inside the tolerance no
    foot slide earns anything. Deploy twin: the resolver snaps table_vector to
    home below the same radius.
    command_name (dp6c): hold target shifted by the commanded-lean displacement —
    see _lean_fwd_shift; the anchor is STILL held at full strength, measured at
    the lean-consistent base pose.

    anchor_1's unbounded quadratic (-30 * d^2) diverged PPO via VALUE-function
    explosion: outlier envs (pushed/walked far) accumulated astronomically
    negative returns that poisoned the critic while the MEDIAN policy still
    looked fine on film (2026-07-25 checkpoint autopsy). This kernel is bounded
    [0,1]: max per-step loss for an outlier = just missing the bonus. Same
    attractor shape — ~free inside sigma (micro-sway), pull beyond, saturates
    far away. Use as a POSITIVE reward (weight ~ +8).
    """
    if not hasattr(env, "spawn_root_xy") or not hasattr(env, "spawn_yaw"):
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    delta = asset.data.root_pos_w[:, :2] - env.spawn_root_xy
    d_lean, fwd_lean = _lean_fwd_shift(env, command_name, lean_height)
    if d_lean is not None:
        delta = delta - d_lean.unsqueeze(-1) * fwd_lean
    if deadband > 0.0:
        dist = torch.norm(delta, dim=-1)
        pos_err2 = torch.clamp(dist - deadband, min=0.0) ** 2
    else:
        pos_err2 = torch.sum(delta * delta, dim=-1)
    fwd = torch.stack([torch.cos(env.spawn_yaw), torch.sin(env.spawn_yaw)], dim=-1)
    if heading_to == "spawn":
        # dp4c heading fix (pushfull rotation bug, 2026-08-06): displaced off
        # home, the BEARING to the anchor disagrees with the table-parallel
        # spawn yaw by up to ~60 deg — the policy turns while returning. The
        # desk requirement is TABLE PARALLELISM: hold the anchor's own
        # orientation (spawn yaw), not the line of sight to it.
        desired_yaw = env.spawn_yaw
    else:
        to_anchor = env.spawn_root_xy + fwd_offset * fwd - asset.data.root_pos_w[:, :2]
        desired_yaw = torch.atan2(to_anchor[:, 1], to_anchor[:, 0])
    q = asset.data.root_quat_w
    yaw = torch.atan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]),
    )
    yaw_err = torch.atan2(torch.sin(desired_yaw - yaw), torch.cos(desired_yaw - yaw))
    err2 = pos_err2 + heading_scale * yaw_err * yaw_err
    return torch.exp(-err2 / (sigma * sigma))


def shoulder_pose_target(
    env: "ManagerBasedRLEnv",
    target_height: float = 1.25,
    target_fwd: float = 0.15,
    heading_from_spawn: bool = True,
    shoulder_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_shoulder_pitch_link"]),
    ankle_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
) -> torch.Tensor:
    """DP2B LEAN (the boss's static pre-tilt): hold the SHOULDER midpoint at a
    forward-and-low target — one body point constrained, everything else free.

    err = (shoulder_mid_z - target_height)^2
        + (forward_offset(shoulder_mid vs ankle_mid, spawn-yaw frame) - target_fwd)^2

    Both measurements are body-configuration-invariant (world height + offset
    from the SUPPORT, in the spawn heading) — NOT torso-frame, which would lean
    with the robot (the tilt-1 frame bug). With shoulders held forward+low and
    CoM-over-feet enforced by not-falling, the pelvis migrates backward and the
    knees soften on their own: physics supplies the counterweight, no pose
    prescription. Pair with flat_orientation_l2 tuned DOWN (~-0.5), not zeroed.
    """
    if not hasattr(env, "spawn_yaw"):
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[shoulder_cfg.name]
    sh_mid = asset.data.body_pos_w[:, shoulder_cfg.body_ids].mean(dim=1)   # (N,3)
    ank_mid = asset.data.body_pos_w[:, ankle_cfg.body_ids].mean(dim=1)
    h_err = sh_mid[:, 2] - target_height
    fwd = torch.stack([torch.cos(env.spawn_yaw), torch.sin(env.spawn_yaw)], dim=-1)
    fwd_off = torch.sum((sh_mid[:, :2] - ank_mid[:, :2]) * fwd, dim=-1)
    f_err = fwd_off - target_fwd
    return h_err * h_err + f_err * f_err


def hand_reach_bonus(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_wrist_yaw_link"]),
) -> torch.Tensor:
    """DP2B TILT2: bounded bonus for the CLOSEST hand approaching the WORLD-frame
    reach point (env.reach_point_w, resampled per episode by
    resample_reach_point). exp(-(min_hand_dist/sigma)^2), in [0,1].

    THE lean gradient: the policy cannot move the arms (externally driven), but
    it can move the shoulders they hang from — when the point is beyond upright
    reach, leaning is the only way to collect. World-frame by construction:
    leaning toward the point genuinely closes the distance (the tilt-1
    torso-frame bug is structurally impossible here).
    """
    if not hasattr(env, "reach_point_w"):
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    hands = asset.data.body_pos_w[:, asset_cfg.body_ids]                   # (N,2,3)
    d = torch.norm(hands - env.reach_point_w.unsqueeze(1), dim=-1)         # (N,2)
    dmin = d.min(dim=-1).values
    return torch.exp(-(dmin / sigma) ** 2)



def _recovery_gate(env, asset, gate_mode: str, gate_speed: float, gate_anchor_dist, push_window_s: float = 2.0) -> torch.Tensor:
    """dp5g: the WHEN of recovery-gated stepping income (feet_gait_recovery,
    foot_clearance_recovery).
      "or"      — legacy: moving (> gate_speed) OR displaced (> gate_anchor_dist).
                  Policy-controllable: stepping moves the base -> gate opens ->
                  stepping is paid (dp5f_gaitpack stampede).
      "transit" — displaced from the anchor only (> gate_anchor_dist): pays the
                  walk-back, silent near home. Costs anchor_hold to trigger.
      "pushwin" — a push happened within push_window_s (env.last_push_t, set by
                  push_by_setting_velocity_stamped) OR a sustained push is active
                  (_sustained_push_clear_time). Privileged, reward-side only —
                  the policy feels the shove through proprioception exactly as on
                  hardware; it just cannot fake the window.
      "prop"    — dp8 (operator 2026-09-09: 'gaiting needs to be active all
                  the time and proportional to the amount it needs to walk'):
                  a FLOAT gate = clamp(dist_from_anchor / gate_anchor_dist, 0, 1).
                  No threshold: 5 cm off pays a quarter of the gait income at
                  gate_anchor_dist 0.20, 20 cm+ pays it in full, at home it is
                  zero (no marching subsidy at rest, Bible rule 1). Callers
                  multiply by .float(), a no-op on a float tensor."""
    if gate_mode == "prop":
        if gate_anchor_dist is None or not hasattr(env, "spawn_root_xy"):
            return torch.zeros(env.num_envs, device=env.device)
        dist = torch.norm(asset.data.root_pos_w[:, :2] - env.spawn_root_xy, dim=-1)
        return (dist / gate_anchor_dist).clamp(0.0, 1.0)
    if gate_mode == "pushwin":
        t = env.episode_length_buf.float() * env.step_dt
        gate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if hasattr(env, "last_push_t"):
            gate = gate | ((t - env.last_push_t) <= push_window_s)
        if hasattr(env, "_sustained_push_clear_time"):
            gate = gate | (env._sustained_push_clear_time > t)
        return gate
    far = None
    if gate_anchor_dist is not None and hasattr(env, "spawn_root_xy"):
        far = torch.norm(asset.data.root_pos_w[:, :2] - env.spawn_root_xy, dim=-1) > gate_anchor_dist
    if gate_mode == "transit":
        return far if far is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    moving = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=-1) > gate_speed
    return moving | far if far is not None else moving


def feet_gait_recovery(
    env: "ManagerBasedRLEnv",
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    gate_speed: float = 0.15,
    gate_anchor_dist: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gate_mode: str = "or",
    push_window_s: float = 2.0,
) -> torch.Tensor:
    """P13: walk's feet_gait adapted for the BALANCE task — same antiphase
    contact-schedule reward, but gated on RECOVERY-IN-PROGRESS (base speed >
    gate_speed) instead of walk's velocity-command gate, which balance does not
    have. Ungated, the free-running phase clock demands one foot airborne half
    the time and INDUCES marching at rest — the exact fidget the balance line
    fights. Gated: at rest the term is silent; the moment the robot is already
    moving (stepping to catch itself), it pays for one-foot-planted antiphase
    stepping. "You may step whenever you must — but when you do, step
    rhythmically: one foot planted, proper swing."

    NOTE (deliberate): NO gait-phase observation is added — the policy cannot
    see the clock (keeps the 87-obs warmstart/deploy contract). It does not
    need to: with antiphase offsets, matching the hidden clock in expectation
    means "exactly one foot in contact, alternating at ~period" (right-foot
    match pays 2/2, wrong-foot 0/2, double-support or double-air 1/2) — the
    gradient favors paced single-support stepping regardless of phase
    alignment. If the pace itself matters later, the phase obs is a p14
    decision (obs change = scratch run).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    leg_phase = torch.cat([(global_phase + o) % 1.0 for o in offset], dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += (~(is_stance ^ is_contact[:, i])).float()

    asset = env.scene[asset_cfg.name]
    recovering = _recovery_gate(env, asset, gate_mode, gate_speed, gate_anchor_dist, push_window_s)
    return reward * recovering.float()


def desk_reach_bonus(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.2,
    command_name: str = "arm_pose_command",
    gate_anchor_dist: float | None = None,
) -> torch.Tensor:
    """dp4c LEAN PROGRAM: bounded bonus for each hand approaching its WISH
    (the pre-resolution world target), gated to DESK-PLANE draws. The arm
    joints are externally commanded, so the only actuator that reduces the
    world-frame hand-to-wish gap is the BODY: shoulder forward = lean. This is
    the tilt2 hand_reach gradient revived on the wish machinery. exp kernel,
    [0,1] per arm, averaged. Zeros when no desk draw is active.

    gate_anchor_dist (dp8, operator 2026-09-10 'gated on anchor distance'): pay
    the reach ONLY while the base is within this radius of its anchor
    (env.spawn_root_xy). Off home the term is silent, so a displaced robot
    walks back first instead of leaning at the target from where it stands —
    the 09-07 'leaned into the table' path. None = legacy, ungated."""
    try:
        cmd = env.command_manager.get_term(command_name)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    total = torch.zeros(env.num_envs, device=env.device)
    for side in ("left", "right"):
        ee = robot.data.body_pos_w[:, cmd.ee_body_idx[side]]
        err2 = torch.sum((ee - cmd.wish_w[side]) ** 2, dim=-1)
        bonus = torch.exp(-err2 / (sigma * sigma))
        active = cmd.desk_wish_mask[side] & ~cmd.default_mode
        total = total + bonus * active.float()
    total = total * 0.5
    if gate_anchor_dist is not None and hasattr(env, "spawn_root_xy"):
        near = torch.norm(robot.data.root_pos_w[:, :2] - env.spawn_root_xy, dim=-1) < gate_anchor_dist
        total = total * near.float()
    return total


def desk_hit_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("desk_contact"),
    force_thr: float = 1.0,
    max_val: float = 10.0,
) -> torch.Tensor:
    """dp4c_deskcol: HAND-vs-DESK strike penalty (filtered contact matrix on
    the desk body vs the wrist links), hinged above force_thr, CLAMPED. Legs
    are deliberately excluded — base_forward_zone owns body-desk spacing; this
    term owns the hand-strike failure mode."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fm = sensor.data.force_matrix_w
    if fm is None:
        return torch.zeros(env.num_envs, device=env.device)
    mag = torch.norm(fm, dim=-1).sum(dim=(-2, -1))
    # NaN/Inf guard (dp5b). clamp() does NOT sanitize NaN — every comparison
    # against NaN is False, so torch.clamp passes it straight through and a
    # single bad contact sample lands in the reward, then the return, then the
    # critic. Same guard the sustained-push sampler has carried since 9142dcd,
    # applied to the READ side of the contact pipe.
    mag = torch.nan_to_num(mag, nan=0.0, posinf=max_val + force_thr, neginf=0.0)
    return (mag - force_thr).clamp(min=0.0, max=max_val)


def heading_l1_from_spawn(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """LINEAR heading deviation from spawn yaw (|dyaw|, radians), wrapped.

    dp4c heading program (2026-08-09): the L2 form's gradient VANISHES near
    zero, so a small residual yaw offset is nearly free and the robot stays
    slightly crooked; L1 keeps a constant restoring gradient at every error
    size. Matched to `heading_l2_from_spawn` at 0.3 rad when weighted -1.5
    against L2's -5.0. Use with a NEGATIVE weight."""
    if not hasattr(env, "spawn_yaw"):
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w
    yaw = torch.atan2(
        2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
        1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]),
    )
    d = yaw - env.spawn_yaw
    d = torch.atan2(torch.sin(d), torch.cos(d))
    return torch.abs(d)


def _desk_draw_active(env, command_name: str = "arm_pose_command"):
    """True per-env while a DESK-PLANE arm draw is active (either hand)."""
    try:
        cmd = env.command_manager.get_term(command_name)
        act = (cmd.desk_wish_mask["left"] | cmd.desk_wish_mask["right"]) & ~cmd.default_mode
        return act
    except Exception:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def upright_bonus_desk_gated(
    env: "ManagerBasedRLEnv",
    std: float = 0.01,
    command_name: str = "arm_pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """dp5_leangate: upright_bonus that PAUSES while the arms work at the desk.

    "Uprightness is the goal when idle, not while reaching." Same kernel as
    upright_bonus (exp(-|g_xy|^2/std^2)) but multiplied by (1 - desk_draw
    active), so peace-time stillness is still paid and a working lean is not
    punished by forfeiture. Bounded [0,1]."""
    asset = env.scene[asset_cfg.name]
    g2 = torch.sum(asset.data.projected_gravity_b[:, :2] ** 2, dim=-1)
    bonus = torch.exp(-g2 / (std ** 2))
    return bonus * (~_desk_draw_active(env, command_name)).float()


# ---------------------------------------------------------------------------
# lm3 (2026-08-12): p13c economy + commanded velocity. The stay-at-spawn and
# torso-stillness terms are correct AT REST and wrong IN MOTION, so they gate
# on the velocity command being ~zero. Pair with events.reanchor_on_stop so
# "spawn" means "where the command last dropped to zero", not birth position.
def _standing_gate(env: "ManagerBasedRLEnv", command_name: str, thr: float = 0.1) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(cmd, dim=1) < thr).float()


def base_pos_xy_hold_standing(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base_pos_xy_l2_from_spawn, active ONLY while commanded to stand.
    With reanchor_on_stop this is station-keeping at the stop point: walk,
    stop, and the p13c hold re-engages wherever the robot is."""
    if not hasattr(env, "spawn_root_xy"):
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    delta = asset.data.root_pos_w[:, :2] - env.spawn_root_xy
    return torch.sum(delta * delta, dim=-1) * _standing_gate(env, command_name)


def body_lin_vel_xy_l2_standing(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
) -> torch.Tensor:
    """torso_lin_vel_xy gated on standing — body_lin_vel_xy_l2 is WORLD-frame
    (verified 2026-08-12), so ungated it taxes commanded walking at -w*v^2."""
    asset = env.scene[asset_cfg.name]
    v = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :2]
    return torch.sum(v * v, dim=-1) * _standing_gate(env, command_name)


def body_ang_vel_l2_standing(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
) -> torch.Tensor:
    """torso_ang_vel gated on standing (commanded turning would pay it)."""
    asset = env.scene[asset_cfg.name]
    w = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], :]
    return torch.sum(w * w, dim=-1) * _standing_gate(env, command_name)


def capture_point_touchdown_standing(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    max_dist: float = 0.5,
    com_offset_b: tuple = (0.0, 0.0),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """lm4e: capture_point_touchdown gated on standing. The capture point is
    the STOP placement (cp = com + v*sqrt(h/g), command-blind); a continuing
    gait must land BEHIND it to keep momentum, so ungated the term is a
    speed-proportional brake on every walking step (~1-2/s at 1 m/s, prime
    suspect for the lm4d ~2/3 forward DC gain) and taxes turning steps.
    Standing keeps its designed anti-push recovery-step job."""
    return capture_point_touchdown_distance(
        env, sensor_cfg, asset_cfg, max_dist, com_offset_b
    ) * _standing_gate(env, command_name)


def joint_target_deviation_l1(
    env: "ManagerBasedRLEnv",
    command_name: str = "arm_pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """lm3 arm discipline: L1 between the arm joints and the arm_pose_command
    TARGETS (not the default pose). In Option II the command drove the arms
    directly; with 27 actions the policy owns them, and this is the attractor
    that keeps the arms on the commanded pose while the legs walk."""
    cmd = env.command_manager.get_command(command_name)
    q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(q - cmd), dim=-1)


def foot_clearance_reward_cmd(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """lm3 FLAG-1 fix (Bible audit, 2026-08-12): foot_clearance_reward GATED on
    a nonzero velocity command.

    The ungated form pays its MAXIMUM at standstill (foot velocity ~0 ->
    tanh ~0 -> exp(0) = 1), while walking WELL scores ~0.74 — measured live on
    lm3_scratch at 10.8/s of 12 while mostly standing. Rule-22 arithmetic: an
    env commanded to walk earned ~the same by disobediently standing
    (clearance 12 + tracking 0) as by walking perfectly (clearance ~9 +
    tracking 3) — an anti-walking subsidy. Correct in the stillness line it
    came from (p13b_clearance); wrong in a walking task. Gate = the in-house
    rewards.py:115 pattern.
    """
    base = foot_clearance_reward(env, asset_cfg, target_height, std, tanh_mult)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    return base * (cmd > 0.1).float()


def arm_gait_swing(
    env: "ManagerBasedRLEnv",
    period: float,
    offset: list[float],
    amplitude: float,
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """lm4c: gait-phase counter-swing income for the shoulder_pitch pair.

    lm4/lm4b left shoulder_pitch deliberately UNTAXED (the counter-swing DOF)
    and the policies used that freedom to park the arms BEHIND the torso as a
    static CoM trim (operator, lm4b previews). This term gives the free DOF a
    JOB instead of a hole: income for tracking default + A*sin(2*pi*phase),
    phase from THE SAME clock as feet_gait_recovery (episode time % period),
    with per-arm offsets — left arm rides the RIGHT leg's phase (offset 0.5),
    right arm the left leg's (0.0): human counter-swing.

    Command-gated exactly like foot_clearance_reward_cmd: pays ONLY when a
    velocity command is active; at standstill the (separate, raised)
    joint_deviation_arms owns the arms. Bounded exp kernel, NaN-safe.
    """
    asset = env.scene[asset_cfg.name]
    global_phase = ((env.episode_length_buf * env.step_dt) % period) / period
    ph = torch.stack([(global_phase + o) % 1.0 for o in offset], dim=-1)
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    target = q0 + amplitude * torch.sin(2.0 * torch.pi * ph)
    err = torch.sum(torch.square(q - target), dim=-1)
    base = torch.exp(-err / std**2)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    return base * (cmd > 0.1).float()


# ---------------------------------------------------------------------------
# lm4d: the crisp-tracking wave (probe-backed, 2026-08-18)
# ---------------------------------------------------------------------------

def track_vel_err_l2(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """lm4d_crisp: the L2-far half of the tracking pair. The narrowed exp
    kernel (std 0.25) goes flat past ~0.5 m/s error; this keeps a gradient
    alive everywhere (design grammar: kernel-near income + L2-far penalty).
    Sums linear-xy and yaw-rate squared errors in the yaw frame."""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    lin_err = torch.sum(
        torch.square(cmd[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=-1
    )
    yaw_err = torch.square(cmd[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return lin_err + yaw_err


def standing_pose_bonus(
    env: "ManagerBasedRLEnv",
    std: float = 0.8,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """lm4d: the firm 'at attention' stance (operator: 'stands like a zombie
    ... I want the standing to be firm and somewhat default like'). Kernel on
    whole-body joint deviation from defaults, paid ONLY at zero command —
    the stance kernels govern foot xy but nothing paid for body posture at
    rest. Gated standing so it cannot fight the gait."""
    asset = env.scene[asset_cfg.name]
    err = torch.sum(
        torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=-1
    )
    base = torch.exp(-err / std**2)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    return base * (cmd <= 0.1).float()


def flat_orientation_split_l2(
    env: "ManagerBasedRLEnv",
    pitch_scale_walking: float = 0.0,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """lm4d_unimpede: tilt discipline SPLIT BY AXIS. Walking requires pitch
    freedom (accel/brake/backward lean) but must never get roll freedom
    (lateral toppling). Roll (gravity-y) is penalized always; pitch
    (gravity-x) fully when standing, scaled by pitch_scale_walking when a
    command is active (0.0 = free lean while walking)."""
    asset = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    roll_sq = torch.square(g[:, 1])
    pitch_sq = torch.square(g[:, 0])
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    walking = (cmd > 0.1).float()
    pitch_w = walking * pitch_scale_walking + (1.0 - walking)
    return roll_sq + pitch_sq * pitch_w


def foot_stance_tracking_standing(
    env: "ManagerBasedRLEnv",
    std: float,
    nominal_foot_pos_b,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """lm4d_unimpede: foot_stance_tracking gated on standing — it is a
    STANDING reward (the lm3 gating pass missed it; measured cost of the miss:
    ~2.25/s forfeited the moment feet leave the nominal spots = half the
    vy/wz refusal arithmetic, probe 2026-08-18)."""
    base = foot_stance_tracking(env, std=std, nominal_foot_pos_b=nominal_foot_pos_b, asset_cfg=asset_cfg)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    return base * (cmd <= 0.1).float()


def stance_bonus_standing(
    env: "ManagerBasedRLEnv",
    std: float,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """lm4d_unimpede: stance_bonus gated on standing (same rationale)."""
    base = stance_bonus(env, std=std, asset_cfg=asset_cfg)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    return base * (cmd <= 0.1).float()


def foot_clearance_recovery(
    env: "ManagerBasedRLEnv",
    std: float = 0.05,
    tanh_mult: float = 2.0,
    target_height: float = 0.12,
    gate_speed: float = 0.15,
    gate_anchor_dist: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
    gate_mode: str = "or",
    push_window_s: float = 2.0,
) -> torch.Tensor:
    """dp5f: swing-clearance income on the DESK line's recovery gate (same
    OR-gate as feet_gait_recovery: moving OR displaced from the anchor). The
    desk line had ZERO clearance income — recovery/transit steps shuffled
    (operator gaiting complaint, dp5e_pain sim2sim). Gated so standing still
    earns nothing (no marching subsidy)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    err = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    swing = torch.tanh(tanh_mult * torch.norm(
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    r = torch.sum(torch.exp(-err / (std * std)) * swing, dim=1)
    gate = _recovery_gate(env, asset, gate_mode, gate_speed, gate_anchor_dist, push_window_s)
    return r * gate.float()


def capture_point_touchdown_anchor(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    max_dist: float = 0.5,
    com_offset_b: tuple = (0.0, 0.0),
    gate_dist: float = 0.3,
) -> torch.Tensor:
    """dp5f: capture_point gated on ANCHOR PROXIMITY — the desk-shaped port of
    the walk line's standing-gate (desk has no velocity command). Near the
    anchor (< gate_dist of spawn) = recovery mode: the cp placement shaper does
    its designed anti-push job. Far = transit mode (walking to a moved anchor):
    the speed-proportional brake is OFF. Falls back to ungated when the spawn
    anchor attr is absent."""
    base = capture_point_touchdown_distance(env, sensor_cfg, asset_cfg, max_dist, com_offset_b)
    if not hasattr(env, "spawn_root_xy"):
        return base
    asset = env.scene[asset_cfg.name]
    near = torch.norm(asset.data.root_pos_w[:, :2] - env.spawn_root_xy, dim=-1) < gate_dist
    return base * near.float()


def lean_track_bonus(
    env: "ManagerBasedRLEnv",
    std: float = 0.05,
    command_name: str = "lean_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """dp6 (colleague's lean command): kernel income for holding the COMMANDED
    pelvis pitch (rad, forward-positive) with roll level. Replaces
    upright_bonus in the Desk6 trunk — upright's sigma-0.01 proj-gravity
    kernel would fight any commanded lean. At cmd=0 this IS an upright bonus
    (sigma in radians)."""
    from isaaclab.utils.math import euler_xyz_from_quat
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    cmd = env.command_manager.get_command(command_name)[:, 0]
    err2 = (pitch - cmd) ** 2 + roll ** 2
    return torch.exp(-err2 / (std * std))


def flat_orientation_lean_l2(
    env: "ManagerBasedRLEnv",
    command_name: str = "lean_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """dp6: the L2-far pair of lean_track_bonus — roll always priced, pitch
    priced against the COMMAND instead of against zero (the walk line's
    flat-split pattern ported to the lean command)."""
    from isaaclab.utils.math import euler_xyz_from_quat
    asset = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    cmd = env.command_manager.get_command(command_name)[:, 0]
    return roll ** 2 + (pitch - cmd) ** 2


def stride_track(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    period: float = 0.75,
    std: float = 0.05,
    min_speed: float = 0.1,
    max_stride: float = 1.2,
    stride_frac: float = 1.0,
    period_slow: float | None = None,
    v_ref: float = 0.8,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
) -> torch.Tensor:
    """lm5d (operator 2026-08-27: "the robot should adapt its feet stride to the
    velocity it is commanded"): at each TOUCHDOWN, kernel on |swing displacement −
    nominal stride|, where nominal = |v_cmd| · period · stride_frac.
    GEOMETRY FIX (2026-08-28, lm5d paid 0.01/s at weight 2 = never in the kernel):
    the measured quantity is ONE foot's liftoff→touchdown displacement. A foot is
    planted during stance, so in its swing it must cover the body's whole travel
    per gait period = v·T (the STRIDE), not v·T/2 (the STEP between opposite
    feet). lm5d's nominal v·T/2 sat ~0.19 m below real strides at 0.5 m/s with
    std 0.05 → r≈0. stride_frac=0.5 reproduces the lm5d (buggy) target. Liftoff xy
    is stored per foot when contact 1→0; on 0→1 the planar displacement since
    liftoff is scored. Gated on |v_cmd| > min_speed so standing pays nothing, and
    the measured stride is clamped (gr law) so a physics spike cannot mint reward.
    Sparse by construction (pays only on touchdown steps)."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0    # (N, 2)
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]                     # (N, 2, 2)
    if not hasattr(env, "_stride_prev_contact"):
        env._stride_prev_contact = torch.ones_like(in_contact)
        env._stride_liftoff_xy = foot_xy.clone()
    if hasattr(env, "reset_buf") and env.reset_buf.any():                          # stale liftoff after a teleport
        rb = env.reset_buf.bool().unsqueeze(-1)
        env._stride_liftoff_xy = torch.where(rb.unsqueeze(-1), foot_xy, env._stride_liftoff_xy)
        env._stride_prev_contact = torch.where(rb, torch.ones_like(in_contact), env._stride_prev_contact)
    liftoff = env._stride_prev_contact & ~in_contact
    touchdown = ~env._stride_prev_contact & in_contact
    env._stride_liftoff_xy = torch.where(liftoff.unsqueeze(-1), foot_xy, env._stride_liftoff_xy)
    stride = torch.norm(foot_xy - env._stride_liftoff_xy, dim=-1).clamp(max=max_stride)  # (N, 2)
    cmd = env.command_manager.get_command(command_name)
    v = torch.norm(cmd[:, :2], dim=-1)                                             # (N,)
    # lm5g (operator 2026-09-02, "should have smaller strides but doesn't calculate
    # the stride lengths that good" below 0.2): a CONSTANT period makes the nominal
    # stride v*0.75 — 7.5 cm at vx 0.1, physically awkward under the fixed 0.4 s
    # feet_air swing. Humans shed speed with BOTH shorter strides and slower
    # cadence; period_slow interpolates the period up as v drops (period at
    # v>=v_ref -> period_slow at v=0), so low speed asks for fewer, decent-length
    # steps instead of a nervous shuffle. period_slow=None = lm5d behaviour.
    if period_slow is not None:
        per = period + (period_slow - period) * (1.0 - v / v_ref).clamp(0.0, 1.0)
    else:
        per = torch.full_like(v, period)
    nominal = (v * per * stride_frac).unsqueeze(1)
    r = torch.exp(-torch.square((stride - nominal) / std)) * touchdown.float()
    env._stride_prev_contact = in_contact.clone()
    return r.sum(dim=1) * (v > min_speed).float()


def stride_symmetry(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    std: float = 0.1,
    min_speed: float = 0.1,
    max_stride: float = 1.2,
    wz_ref: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
) -> torch.Tensor:
    """lm5f (2026-08-31): left/right stride EQUALITY. stride_track pays each foot
    against the nominal INDEPENDENTLY — one long stride plus a catch-up step
    scores like a balanced passing gait, and at low vx (short nominal) the
    catch-up 'step-to' gait is the cheap solution (lm5e sim2sim: "placing one
    foot forward and then placing the other next to it"; the FRESH-learned
    backwards gait, with no lineage habits, came out perfectly symmetric —
    the asymmetry is a warmstart local optimum, so this term prices it
    directly). On each touchdown: kernel on |this foot's completed stride −
    the OTHER foot's last completed stride|. Fades with the commanded yaw rate
    (turning legitimately strides the outer foot longer, same fade as
    joint_deviation_l1_turn_gated); gated on |v_cmd| > min_speed; strides
    clamped (gr law); pays nothing until both feet have one completed stride
    (post-reset). Sparse by construction, same bookkeeping as stride_track but
    on its OWN buffers so either term can run alone."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0    # (N, 2)
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]                     # (N, 2, 2)
    if not hasattr(env, "_ssym_prev_contact"):
        env._ssym_prev_contact = torch.ones_like(in_contact)
        env._ssym_liftoff_xy = foot_xy.clone()
        env._ssym_last_len = torch.zeros_like(foot_xy[..., 0])                     # (N, 2)
        env._ssym_valid = torch.zeros_like(in_contact)                             # completed a stride yet?
    if hasattr(env, "reset_buf") and env.reset_buf.any():
        rb = env.reset_buf.bool().unsqueeze(-1)
        env._ssym_liftoff_xy = torch.where(rb.unsqueeze(-1), foot_xy, env._ssym_liftoff_xy)
        env._ssym_prev_contact = torch.where(rb, torch.ones_like(in_contact), env._ssym_prev_contact)
        env._ssym_valid = torch.where(rb, torch.zeros_like(env._ssym_valid), env._ssym_valid)
    liftoff = env._ssym_prev_contact & ~in_contact
    touchdown = ~env._ssym_prev_contact & in_contact
    env._ssym_liftoff_xy = torch.where(liftoff.unsqueeze(-1), foot_xy, env._ssym_liftoff_xy)
    stride = torch.norm(foot_xy - env._ssym_liftoff_xy, dim=-1).clamp(max=max_stride)  # (N, 2)
    env._ssym_last_len = torch.where(touchdown, stride, env._ssym_last_len)
    env._ssym_valid = env._ssym_valid | touchdown
    other_len = env._ssym_last_len.flip(-1)
    other_ok = env._ssym_valid.flip(-1)
    r = torch.exp(-torch.square((stride - other_len) / std)) * touchdown.float() * other_ok.float()
    env._ssym_prev_contact = in_contact.clone()
    cmd = env.command_manager.get_command(command_name)
    v = torch.norm(cmd[:, :2], dim=-1)
    turn_fade = (1.0 - (cmd[:, 2].abs() / wz_ref).clamp(max=1.0))
    return r.sum(dim=1) * (v > min_speed).float() * turn_fade


def base_height_lean_l2(
    env: "ManagerBasedRLEnv",
    target_height: float = 0.87,
    command_name: str = "lean_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """dp6c: base_height_l2 with the target following the commanded lean —
    a pelvis pitched theta about the ankles sits at ~target*cos(theta); the flat
    -15 * (0.87 - z)^2 taxed every commanded lean for its unavoidable ~5 cm drop."""
    asset = env.scene[asset_cfg.name]
    theta = env.command_manager.get_command(command_name)[:, 0]
    target = target_height * torch.cos(theta)
    return torch.square(asset.data.root_pos_w[:, 2] - target)


def joint_deviation_l1_lean_gated(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "lean_command",
    lean_ref: float = 0.35,
) -> torch.Tensor:
    """dp6c: joint_deviation_l1 fading with the commanded |lean| (the lm5e
    turn-gated pattern) — the hip-pitch excursion a commanded lean NEEDS must
    not be taxed by the hip evictor; at cmd 0 the full tax is back."""
    asset = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    theta = env.command_manager.get_command(command_name)[:, 0].abs()
    fade = 1.0 - (theta / lean_ref).clamp(max=1.0)
    return torch.sum(dev.abs(), dim=-1) * fade


def joint_deviation_l1_turn_gated(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    wz_ref: float = 0.5,
) -> torch.Tensor:
    """lm5e_turn (2026-08-28 yaw audit): joint_deviation_l1 whose weight fades with
    the commanded yaw rate — x(1 - min(|wz_cmd|/wz_ref, 1)). Rationale: the lm4e
    hip EVICTOR (joint_deviation_hips -2.0 on hip roll+yaw) exists to pull the
    parked hip yaw off its -0.43 stop when standing/walking straight; during a
    commanded turn the same term taxes exactly the hip-yaw excursion turning
    needs (~0.3 rad on both hips = -1.2/step against a 5/step tracking ceiling).
    Standing and straight walking keep the full evictor; a full-rate turn frees it."""
    asset: Articulation = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    wz = env.command_manager.get_command(command_name)[:, 2].abs()
    gate = (1.0 - (wz / wz_ref).clamp(max=1.0))
    return torch.sum(dev.abs(), dim=1) * gate


def shoulder_roll_swing(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    min_speed: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"]),
) -> torch.Tensor:
    """lm5g (operator 2026-09-02: "the arm swing seems to [be] from side to side
    instead of pitched and in anti-phase"): PENALTY on lateral arm oscillation —
    sum qd^2 over the shoulder roll/yaw joints, gated to walking (|v_cmd| >
    min_speed) so standing arm posture stays unpriced. Gated penalty is safe
    (Bible law 1 forbids gated INCOME + ungated penalties, not this)."""
    asset: Articulation = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    cmd = env.command_manager.get_command(command_name)
    walking = (torch.norm(cmd[:, :2], dim=-1) > min_speed).float()
    return torch.sum(torch.square(qd), dim=-1) * walking


def arm_antiphase(
    env: "ManagerBasedRLEnv",
    command_name: str = "base_velocity",
    min_speed: float = 0.15,
    scale: float = 2.0,
    arm_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint", "right_shoulder_pitch_joint"]),
    hip_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"]),
) -> torch.Tensor:
    """lm5g: INCOME for the human arm swing — each shoulder PITCH velocity
    anti-phase with the SAME-side hip pitch (left arm forward while left leg
    back). Kernel = relu(tanh(-qd_shoulder * qd_hip / scale)) per side, mean of
    the two sides, walking-gated. Smooth local gradient from any phase (Bible
    law 12); pays 0 (never negative) when in-phase, so it shapes rather than
    fights — drop-safe if it degrades the walk."""
    asset: Articulation = env.scene[arm_cfg.name]
    qd_arm = asset.data.joint_vel[:, arm_cfg.joint_ids]                       # (N, 2) L,R
    qd_hip = asset.data.joint_vel[:, hip_cfg.joint_ids]                       # (N, 2) L,R
    anti = torch.relu(torch.tanh(-(qd_arm * qd_hip) / scale))                 # (N, 2)
    cmd = env.command_manager.get_command(command_name)
    walking = (torch.norm(cmd[:, :2], dim=-1) > min_speed).float()
    return anti.mean(dim=-1) * walking
