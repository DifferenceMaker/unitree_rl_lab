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
) -> torch.Tensor:
    """Digit-style stepping regularizer (Oregon State, arXiv 2404.19173).

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
    return torch.sum(touchdowns, dim=1)


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
    return torch.sum(delta ** 2, dim=-1)


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
