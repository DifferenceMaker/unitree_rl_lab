"""MDP terms for the Unitree-H1_2-LieDown task (safety lie-down v1).

Design sources (paper survey 2026-07-28, see memory liedown-policy-paper-survey):
- HoST (arXiv 2502.08378): height-stage-gated reward groups, post-task
  "stay quiet" group, joint-vel/smoothness regularization, β action bound
  (implemented via RelativeJointPositionActionCfg in the env cfg, not here).
- FIRM (arXiv 2511.07407): damage triplet — body collision force, momentum
  change m·a, contact-force yank ‖ΔF‖² — and minimal terminations (velocity
  limits only; NO height/contact terminations: lying IS the goal state).
- SafeFall (arXiv 2511.18590): per-body-group contact-force thresholds
  (protect torso/head hard, allow pelvis/back cheap at low force).

Sign convention for supine: base x-axis points skyward when lying on the
back, so projected_gravity_b ≈ (-1, 0, 0). Forward faceplant (prone) is
projected_gravity_b[0] → +1 and is a termination.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
Observations (critic / privileged).
"""


def root_height_obs(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root height above ground (flat terrain assumed). Privileged — real
    robot has no absolute height estimate once non-foot bodies touch down."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def body_contact_flags(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 5.0
) -> torch.Tensor:
    """Binary contact state per selected body (privileged critic obs).

    Gives the critic the contact-stage information (which support surfaces
    are established) without requiring a stage state machine.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return (forces.norm(dim=-1) > threshold).float()


"""
Task rewards.
"""


def height_descent_progress(
    env: ManagerBasedRLEnv,
    target_height: float,
    start_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Linear progress from standing height to lying height, in [0, 1].

    Saturates at target_height so there is no incentive to burrow. Monotone
    everywhere between start and target — the dense "go down" gradient
    (HoST's task group, reversed). Speed of descent is NOT rewarded; the
    slowness levers (descent_rate_limit, β bound, damage triplet) shape how
    the height is given up.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    progress = (start_height - h) / (start_height - target_height)
    return progress.clamp(0.0, 1.0)


def lying_orientation_exp(
    env: ManagerBasedRLEnv,
    target_gravity_b: list[float],
    std: float = 0.7,
    gate_height: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exp kernel on projected-gravity error vs the target lying orientation,
    gated to activate only below gate_height (stage gate, HoST-style).

    For supine target_gravity_b = (-1, 0, 0). Gating prevents the term from
    fighting the standing→squat phase where the base must stay upright.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.tensor(target_gravity_b, device=env.device)
    err_sq = torch.sum(
        torch.square(asset.data.projected_gravity_b - target), dim=-1
    )
    gate = (asset.data.root_pos_w[:, 2] < gate_height).float()
    return torch.exp(-err_sq / std**2) * gate


def quiet_lying_bonus(
    env: ManagerBasedRLEnv,
    gate_height: float = 0.35,
    gate_gravity_x: float = -0.6,
    std_lin: float = 0.10,
    std_ang: float = 0.30,
    std_jvel: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Post-task group (HoST reversed): once low AND supine-ish, bonus for
    being completely still — base lin/ang velocity and joint velocities near
    zero. This is what makes the policy END calm instead of thrashing, and
    it dominates the reward budget once reached (weight it highest).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gate = (
        (asset.data.root_pos_w[:, 2] < gate_height)
        & (asset.data.projected_gravity_b[:, 0] < gate_gravity_x)
    ).float()
    lin_sq = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=-1)
    ang_sq = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=-1)
    jvel_sq = torch.mean(torch.square(asset.data.joint_vel), dim=-1)
    bonus = torch.exp(
        -(lin_sq / std_lin**2 + ang_sq / std_ang**2 + jvel_sq / std_jvel**2)
    )
    return bonus * gate


"""
Safety rewards (all return positive magnitudes — use NEGATIVE weights).
"""


def descent_rate_limit(
    env: ManagerBasedRLEnv,
    max_down_speed: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Quadratic penalty on downward base velocity beyond max_down_speed.

    The explicit "slow" governor: free until -max_down_speed m/s, then
    quadratic. Upward velocity is never penalized (recovery moves are fine).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    v_down_excess = (-asset.data.root_lin_vel_w[:, 2] - max_down_speed).clamp(min=0.0)
    return torch.square(v_down_excess)


def contact_force_above_threshold(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Sum of contact-force magnitude EXCESS above threshold over the
    selected bodies (SafeFall-style, minus their per-body gravity share —
    we approximate it with the flat threshold).

    Instantiate once per body group with group-specific threshold + weight:
    resting load stays free, impacts and hard leaning get penalized.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    return (forces - threshold).clamp(min=0.0).sum(dim=-1)


def contact_force_yank(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """FIRM's "body yank": ‖ΔF‖² between the two most recent sensor samples,
    summed over bodies. Penalizes force SPIKES (slams) while leaving steady
    resting load free — the cleanest "soft touchdown" signal.

    Requires ContactSensorCfg(history_length >= 2).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    hist = sensor.data.net_forces_w_history  # (N, T, B, 3), index 0 = newest
    body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    delta = hist[:, 0, body_ids, :] - hist[:, 1, body_ids, :]
    return torch.sum(torch.square(delta), dim=(-2, -1))


def body_momentum_rate(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """FIRM's "momentum change": Σ_b m_b·‖a_b‖ over all bodies. Large during
    free-fall segments and abrupt arrests; small during quasi-static
    lowering. Complements yank (which only sees contact events)."""
    asset: Articulation = env.scene[asset_cfg.name]
    masses = getattr(env, "_liedown_body_masses", None)
    if masses is None:
        masses = asset.data.default_mass.to(env.device)
        env._liedown_body_masses = masses
    acc = asset.data.body_acc_w[:, :, 0:3]
    return torch.sum(masses * acc.norm(dim=-1), dim=-1)


def action_smoothness_2nd(env: ManagerBasedRLEnv) -> torch.Tensor:
    """HoST's second-difference smoothness ‖a_t − 2a_{t−1} + a_{t−2}‖².

    Keeps an a_{t−2} buffer on the env (updated in-place here; reward terms
    run exactly once per step). The one-step transient after env resets is
    negligible against 4096 envs × 1000-step episodes.
    """
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    a_prev2 = getattr(env, "_liedown_prev_prev_action", None)
    if a_prev2 is None:
        a_prev2 = a_prev.clone()
    penalty = torch.sum(torch.square(a - 2.0 * a_prev + a_prev2), dim=-1)
    env._liedown_prev_prev_action = a_prev.clone()
    return penalty


"""
Terminations.
"""


def root_velocity_limit(
    env: ManagerBasedRLEnv,
    max_lin_vel: float = 2.5,
    max_ang_vel: float = 6.0,
    grace_period_s: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base is genuinely ballistic (thrown/crashed) —
    FIRM's minimal-termination recipe. A controlled lie-down never gets
    close to these rates; anything above them is a failed episode.

    grace_period_s masks the first steps after reset: PhysX depenetration
    at spawn produces a 1–4 step root-velocity spike (angular is uncapped)
    that otherwise terminates 100% of episodes at ~3.5 steps.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    lin = asset.data.root_lin_vel_w.norm(dim=-1) > max_lin_vel
    ang = asset.data.root_ang_vel_w.norm(dim=-1) > max_ang_vel
    grace_steps = int(grace_period_s / (env.cfg.sim.dt * env.cfg.decimation))
    past_grace = env.episode_length_buf > grace_steps
    return (lin | ang) & past_grace


def prone_orientation(
    env: ManagerBasedRLEnv,
    gravity_x_threshold: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate on forward faceplant: base x-axis pointing DOWN means the
    robot pitched forward past ~45–50° (chest/face toward ground). The head
    and cameras live on the torso front — this is the one orientation the
    task must never visit. Supine (gravity_x → -1) is the goal and stays
    unterminated."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b[:, 0] > gravity_x_threshold


def controlled_descent_bonus(
    env: ManagerBasedRLEnv,
    target_speed: float = 0.20,
    std: float = 0.12,
    min_height: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """SD2: the POSITIVE twin of descent_rate_limit.

    descent_rate_limit only punishes going down too fast, so "not descending at
    all" scores identically to "descending perfectly" — there was no gradient
    that made a slow, deliberate descent better than standing still. This pays
    an exp kernel for keeping the downward base speed NEAR target_speed while
    still above min_height (i.e. while there is descending left to do):

        exp(-((v_down - target_speed) / std)^2),  gated on pelvis > min_height

    v_down is positive going down. Standing still (v_down=0) scores
    exp(-(0.20/0.12)^2) = 0.06; descending at 0.20 m/s scores 1.0. Above
    min_height the gate closes so the robot is not asked to keep sinking once
    it is already down — quiet_lying takes over there.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    v_down = -asset.data.root_lin_vel_w[:, 2]
    gate = (asset.data.root_pos_w[:, 2] > min_height).float()
    return torch.exp(-(((v_down - target_speed) / std) ** 2)) * gate


def settled_supine_bonus(
    env: ManagerBasedRLEnv,
    height_thr: float = 0.30,
    gravity_x_thr: float = -0.7,
    lin_thr: float = 0.15,
    ang_thr: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """SD2 terminal-shaped SUCCESS signal: 1.0 while the robot is genuinely
    down, on its back, and stopped — the 'you have finished the job' payment.

    Distinct from quiet_lying (a soft exp kernel that pays partial credit for
    being nearly-still): this is a hard indicator, so a policy that lingers
    half-supine gets nothing from it. It is what replaces an alive bonus for a
    do-one-thing-then-stop task: the reward for existing is conditional on
    having COMPLETED the descent, not on time survived.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    low = asset.data.root_pos_w[:, 2] < height_thr
    supine = asset.data.projected_gravity_b[:, 0] < gravity_x_thr
    still = (asset.data.root_lin_vel_w.norm(dim=-1) < lin_thr) & (
        asset.data.root_ang_vel_w.norm(dim=-1) < ang_thr
    )
    return (low & supine & still).float()
