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


def seed_descent_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    seed_fraction: float = 0.5,
    start_height: float = 0.95,
    end_height: float = 0.20,
    end_pitch_deg: float = -80.0,
    anneal_steps: int = 6_000_000,
    u_lo_start: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """SD2B: BACKWARD-CHAINING goal seeding — the fix for "the two biggest
    rewards read 0.0000 forever". A random seed_fraction of resets does NOT
    start standing: it starts PART-WAY down the descent path, at progress
    u in [u_lo, 1] (u=1 = fully supine on the ground). Seeded envs experience
    quiet_lying/lying_orientation/settled pay from step 0, the value function
    learns the goal state is valuable, and that value propagates backward to
    the standing starts.

    "Progressively raise the robot up": u_lo anneals from u_lo_start down to 0
    over anneal_steps common steps — early training seeds cluster NEAR the
    goal, later training seeds span the whole path back up to standing.

    Pose along the path: pelvis height lerp(start_height -> end_height),
    pitch lerp(0 -> end_pitch_deg) about +y (pitching BACKWARD: base x-axis
    tips upward -> supine, gravity_b x -> -1). JOINTS interpolate default
    crouch -> flat (zeros: straight legs, arms at sides) with the same u, and
    the root gets +0.05 m clearance so nothing spawns intersecting the ground.
    First smoke without this: the crouch legs punched through the floor at
    high u, PhysX depenetration launched every robot, and 100% of episodes
    died ballistic at ~26 steps = the 0.5 s spawn grace + 1.

    Runs AFTER reset_base/reset_robot_joints in the event list (overwrites the
    root state of the seeded subset only).
    """
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    if n == 0:
        return
    dev = env.device
    seeded = torch.rand(n, device=dev) < seed_fraction
    ids = env_ids[seeded]
    if len(ids) == 0:
        return
    # annealed lower bound: near-goal early, whole-path later
    frac = min(1.0, float(env.common_step_counter) / float(anneal_steps))
    u_lo = u_lo_start * (1.0 - frac)
    u = u_lo + torch.rand(len(ids), device=dev) * (1.0 - u_lo)

    root = asset.data.default_root_state[ids].clone()
    root[:, :3] = env.scene.env_origins[ids]
    root[:, 2] = start_height + u * (end_height - start_height) + 0.05
    pitch = torch.deg2rad(torch.tensor(end_pitch_deg, device=dev)) * u
    half = pitch / 2.0
    root[:, 3] = torch.cos(half)   # w
    root[:, 4] = 0.0               # x
    root[:, 5] = torch.sin(half)   # y  (pitch about +y; negative angle = backward)
    root[:, 6] = 0.0               # z
    root[:, 7:] = 0.0              # zero velocity
    asset.write_root_pose_to_sim(root[:, :7], env_ids=ids)
    asset.write_root_velocity_to_sim(root[:, 7:], env_ids=ids)
    # joints: default crouch at u=0 -> flat (zeros) at u=1
    jp = asset.data.default_joint_pos[ids] * (1.0 - u.unsqueeze(-1))
    jv = torch.zeros_like(jp)
    asset.write_joint_state_to_sim(jp, jv, env_ids=ids)


def settled_success(
    env: ManagerBasedRLEnv,
    height_thr: float = 0.30,
    gravity_x_thr: float = -0.7,
    lin_thr: float = 0.15,
    ang_thr: float = 0.5,
    sustain_steps: int = 25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """SD2B SUCCESS termination: low AND supine AND stopped, SUSTAINED for
    sustain_steps consecutive control steps (0.5 s at 50 Hz — a settle, not a
    bounce through the target region). Ends the episode so the return is
    bounded and the optimum is to FINISH, not to linger collecting per-step
    bonuses. Pair with an is_terminated_term success bonus."""
    asset = env.scene[asset_cfg.name]
    if not hasattr(env, "liedown_settle_count"):
        env.liedown_settle_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    ok = (
        (asset.data.root_pos_w[:, 2] < height_thr)
        & (asset.data.projected_gravity_b[:, 0] < gravity_x_thr)
        & (asset.data.root_lin_vel_w.norm(dim=-1) < lin_thr)
        & (asset.data.root_ang_vel_w.norm(dim=-1) < ang_thr)
    )
    env.liedown_settle_count = torch.where(
        ok, env.liedown_settle_count + 1, torch.zeros_like(env.liedown_settle_count)
    )
    return env.liedown_settle_count >= sustain_steps


def prone_excess(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """SD2B: hinged per-step penalty on FORWARD pitch (toward prone) — replaces
    the faceplant TERMINATION. sd1/sd2 proved the termination was a free exit
    from accumulated cost: the policy reached it in 9 steps, 100% of episodes.
    As a bounded per-step cost with no exit attached, tipping forward is just
    a bad place to be that must be lived with — recovering toward supine is
    the only way to stop paying. relu(gravity_b_x - threshold), max ~0.5."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return (asset.data.projected_gravity_b[:, 0] - threshold).clamp(min=0.0)
