"""MDP pieces for the Inspire FTP tactile grasp task (gr line, 2026-07-28).

Design contract (agreed 2026-07-28):
  * ACTIONS = the 6 real hand motors ONLY, in the REAL SDK order
    (angle_set[0..5] = little, ring, middle, index, thumb_bend, thumb_rot).
    The arm never moves — deploy-side the arm is IK-resolved separately.
  * Mimic coupling lives in the ACTION MAPPING, not in physics: the URDF
    importer drops <mimic>, so followers are normal driven joints commanded
    at multiplier x driver (the real linkage enforces this in steel).
  * TACTILE = 17 pad ContactSensor force magnitudes, matching the 17 regions
    of the real inspire_hand_touch DDS message one-to-one. Deploy pools each
    taxel array to one scalar -> identical observation.
  * Blind actor (tactile + proprioception), sighted critic (cube pose/vel,
    privileged) — the HuB asymmetric pattern.
  * The TEST is support removal: a kinematic platform under the cube retracts
    at a random time; a held cube stays at the palm, an uncaged one falls.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# The gr5b arm-in-the-loop chain (Arm7 variant), shoulder -> wrist. Default for
# the gr5c arm-shaping terms; the Wrist3 variant passes its own 3-joint list.
# SIDE-AGNOSTIC names (right-hand era, 2026-08-24): every grasp scene carries
# exactly ONE hand, so "(left|right)_" regexes resolve to that hand's joints —
# the same code runs the left asset (inspire_hand_arm7) and the right one
# (inspire_hand_arm7_right) with zero behavioral change for existing left
# tasks (regex matches the only hand present).
_SIDE = "(left|right)"
ARM7_JOINTS = [
    f"{_SIDE}_shoulder_pitch_joint",
    f"{_SIDE}_shoulder_roll_joint",
    f"{_SIDE}_shoulder_yaw_joint",
    f"{_SIDE}_elbow_joint",
    f"{_SIDE}_wrist_roll_joint",
    f"{_SIDE}_wrist_pitch_joint",
    f"{_SIDE}_wrist_yaw_joint",
]
WRIST3_JOINTS = ARM7_JOINTS[4:]

# Real Inspire SDK motor order: angle_set[0..5]. Drivers in the URDF.
DRIVER_JOINTS = [
    f"{_SIDE}_little_1_joint",   # 0
    f"{_SIDE}_ring_1_joint",     # 1
    f"{_SIDE}_middle_1_joint",   # 2
    f"{_SIDE}_index_1_joint",    # 3
    f"{_SIDE}_thumb_2_joint",    # 4  thumb bend
    f"{_SIDE}_thumb_1_joint",    # 5  thumb rotation/yaw
]
# follower -> (driver, multiplier), keyed by the SIDE-STRIPPED suffix (the
# names find_joints returns are RESOLVED, e.g. right_little_2_joint — a
# regex-keyed dict would KeyError). Thumb chain resolved to the bend driver.
_COUPLING_BY_SUFFIX = {
    "little_2_joint": ("little_1_joint", 1.0843),
    "ring_2_joint": ("ring_1_joint", 1.0843),
    "middle_2_joint": ("middle_1_joint", 1.0843),
    "index_2_joint": ("index_1_joint", 1.0843),
    "thumb_3_joint": ("thumb_2_joint", 0.8024),
    "thumb_4_joint": ("thumb_2_joint", 0.8024 * 0.9487),
}
FOLLOWER_COUPLING = {f"{_SIDE}_{k}": (f"{_SIDE}_{v[0]}", v[1])
                     for k, v in _COUPLING_BY_SUFFIX.items()}


# re-exported so grasp jobs can attach the geometric in-slab termination by name
# (gr7c: "the hand teleports inside the table" — same tunneling family as dp6b)
from unitree_rl_lab.tasks.locomotion.mdp.terminations import desk_penetration  # noqa: F401, E402


def _strip_side(name: str) -> str:
    """left_little_2_joint / right_little_2_joint -> little_2_joint."""
    return name.split("_", 1)[1]


PALM_BODY = f"{_SIDE}_palm_force_sensor"
THUMB_PAD_SUBSTR = "thumb_force_sensor"


# ---------------------------------------------------------------------------
# Action term: 6 motors -> 12 joint targets (couplings applied)
# ---------------------------------------------------------------------------
class CoupledFingerAction(ActionTerm):
    """action a in [-1, 1]^6 -> driver target = lo + (a+1)/2 * (hi-lo);
    follower target = multiplier * driver target (clamped to follower limits).

    Deploy mapping is affine: real angle_set = 1000 * (1 - (a+1)/2) per motor
    (Inspire convention: 1000 = open, 0 = closed) — one linear remap, no
    learned component.
    """

    def __init__(self, cfg: CoupledFingerActionCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.asset_name]
        d_ids, d_names = asset.find_joints(DRIVER_JOINTS, preserve_order=True)
        self._driver_ids = torch.tensor(d_ids, device=self.device)
        f_names = list(FOLLOWER_COUPLING.keys())
        f_ids, f_found = asset.find_joints(f_names, preserve_order=True)
        self._follower_ids = torch.tensor(f_ids, device=self.device)
        # follower i is driven by driver at index _f2d[i] with multiplier _fmul[i]
        # f_found / d_names are RESOLVED joint names (side-specific); map
        # follower -> driver via side-stripped suffixes.
        d_suffix = [_strip_side(n) for n in d_names]
        self._f2d = torch.tensor(
            [d_suffix.index(_COUPLING_BY_SUFFIX[_strip_side(n)][0]) for n in f_found],
            device=self.device,
        )
        self._fmul = torch.tensor(
            [_COUPLING_BY_SUFFIX[_strip_side(n)][1] for n in f_found], device=self.device
        )
        limits = asset.data.soft_joint_pos_limits  # (N, J, 2)
        self._d_lo = limits[:, d_ids, 0]
        self._d_hi = limits[:, d_ids, 1]
        self._f_lo = limits[:, f_ids, 0]
        self._f_hi = limits[:, f_ids, 1]
        self._raw = torch.zeros(self.num_envs, 6, device=self.device)
        self._targets_d = torch.zeros_like(self._raw)
        self._targets_f = torch.zeros(self.num_envs, len(f_ids), device=self.device)
        self._asset: Articulation = asset
        # deploy exporter contract: the policy's joints, in action order
        self._joint_ids = d_ids

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._targets_d

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        a01 = (actions.clamp(-1.0, 1.0) + 1.0) * 0.5
        self._targets_d = self._d_lo + a01 * (self._d_hi - self._d_lo)

    def apply_actions(self):
        # gr3 COUPLING FIX: follower target = ratio * driver MEASURED position,
        # re-evaluated every control step — not ratio * driver TARGET.
        # The real 4-bar couples measured states: when the cube blocks the
        # proximal, the distal stops WITH it. Target-coupling cannot express
        # that: with soft followers the distal sagged up to 68deg past the
        # linkage ratio under load; with stiff followers it OVERSHOT (the
        # blocked driver never reaches its target, so ratio*target is a pose
        # the mechanism never visits — probe went 68 -> 87deg). Measured-state
        # coupling + stiff follower drives (300/2) tracks the linkage against
        # whatever the driver actually did. One-step lag; one-way (follower
        # load does not back-drive the driver — positionally faithful, torque
        # path approximate).
        d_meas = self._asset.data.joint_pos[:, self._driver_ids]
        f_meas = self._asset.data.joint_pos[:, self._follower_ids]
        # ...and BACK-DRIVE, the other half of the linkage: when the cube blocks
        # the DISTAL segment (little/ring reach it distal-first), the real 4-bar
        # stalls the motor — the driver stops too. One-way coupling left 65-70deg
        # of driver run-past on exactly those fingers. Clamp each driver's
        # effective target to what its followers' measured state allows
        # (min over its followers of f_meas/ratio) + a small tracking margin.
        # margin 0.08 rad: probe-tuned. 0.03 chokes the driver against the
        # follower's tracking lag and limit-cycles (index dev bounced 18->39deg);
        # 0.08 is the stable optimum (worst finger 68->21deg, thumb 6deg).
        allowed = f_meas / self._fmul + 0.08
        d_cap = torch.full_like(self._targets_d, float("inf"))
        d_cap.scatter_reduce_(1, self._f2d.unsqueeze(0).expand_as(allowed), allowed, reduce="amin")
        d_eff = torch.minimum(self._targets_d, d_cap)
        self._asset.set_joint_position_target(d_eff, joint_ids=self._driver_ids)
        targets_f = (d_meas[:, self._f2d] * self._fmul).clamp(self._f_lo, self._f_hi)
        self._asset.set_joint_position_target(targets_f, joint_ids=self._follower_ids)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._raw[:] = 0.0
        else:
            self._raw[env_ids] = 0.0


@configclass
class CoupledFingerActionCfg(ActionTermCfg):
    class_type: type = CoupledFingerAction
    asset_name: str = "robot"
    # introspected by export_deploy_cfg (deploy.yaml); the term maps a in [-1,1]
    # to the joint range itself, so scale is nominal
    scale: float = 1.0
    clip = None


# ---------------------------------------------------------------------------
# Buffers (lazy) + helpers
# ---------------------------------------------------------------------------
def _buffers(env: "ManagerBasedRLEnv"):
    if not hasattr(env, "grasp_retract_t"):
        env.grasp_retract_t = torch.full((env.num_envs,), 1.0e9, device=env.device)
        env.grasp_retracted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env


def _pad_force_mags(env: "ManagerBasedRLEnv", sensor_name: str = "pad_sensor") -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    return sensor.data.net_forces_w.norm(dim=-1)  # (N, 17)


def _pad_masks(env: "ManagerBasedRLEnv", sensor_name: str = "pad_sensor"):
    if not hasattr(env, "grasp_thumb_mask"):
        names = env.scene.sensors[sensor_name].body_names
        thumb = torch.tensor([THUMB_PAD_SUBSTR in n for n in names], device=env.device)
        env.grasp_thumb_mask = thumb
        env.grasp_finger_mask = ~thumb & torch.tensor(
            ["palm_force" not in n for n in names], device=env.device
        )
    return env.grasp_thumb_mask, env.grasp_finger_mask


def _palm_pose(env: "ManagerBasedRLEnv"):
    asset: Articulation = env.scene["robot"]
    if not hasattr(env, "grasp_palm_body_id"):
        ids, _ = asset.find_bodies([PALM_BODY])
        env.grasp_palm_body_id = ids[0]
    bid = env.grasp_palm_body_id
    return asset.data.body_pos_w[:, bid], asset.data.body_quat_w[:, bid]


def _wrist_pose(env: "ManagerBasedRLEnv"):
    """World pose of the wrist_yaw link — the rigid frame the gr8 object
    offset is expressed in (see reset_scene_grasp object_offset_wrist_frame)."""
    asset: Articulation = env.scene["robot"]
    if not hasattr(env, "grasp_wrist_body_id"):
        ids, _ = asset.find_bodies([f"{_SIDE}_wrist_yaw_link"])
        env.grasp_wrist_body_id = ids[0]
    bid = env.grasp_wrist_body_id
    return asset.data.body_pos_w[:, bid], asset.data.body_quat_w[:, bid]


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
def pad_forces_log(env: "ManagerBasedRLEnv", scale: float = 0.5) -> torch.Tensor:
    """17 tactile scalars: log1p(|F|) * scale. Deploy analog: log1p(pooled
    taxel value * calib) — the log squashes the (uncalibrated) magnitude scale,
    the noise/DR covers the residual mismatch."""
    return torch.log1p(_pad_force_mags(env)) * scale


def driver_joint_pos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """6 driver joint positions normalized to [0,1] inside their limits —
    exactly what the real hand reports back (pos_act/angle_act rescaled)."""
    asset: Articulation = env.scene["robot"]
    if not hasattr(env, "grasp_driver_ids"):
        ids, _ = asset.find_joints(DRIVER_JOINTS, preserve_order=True)
        env.grasp_driver_ids = torch.tensor(ids, device=env.device)
        lim = asset.data.soft_joint_pos_limits
        env.grasp_driver_lo = lim[:, ids, 0]
        env.grasp_driver_span = (lim[:, ids, 1] - lim[:, ids, 0]).clamp(min=1e-6)
    jp = asset.data.joint_pos[:, env.grasp_driver_ids]
    return (jp - env.grasp_driver_lo) / env.grasp_driver_span


def cube_pos_in_palm(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """PRIVILEGED (critic only): cube center in the palm-pad frame."""
    cube: RigidObject = env.scene["cube"]
    p_pos, p_quat = _palm_pose(env)
    return math_utils.quat_apply_inverse(p_quat, cube.data.root_pos_w - p_pos)


def cube_vel_in_palm(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """PRIVILEGED: cube lin+ang velocity in the palm frame (6)."""
    cube: RigidObject = env.scene["cube"]
    _, p_quat = _palm_pose(env)
    lin = math_utils.quat_apply_inverse(p_quat, cube.data.root_lin_vel_w)
    ang = math_utils.quat_apply_inverse(p_quat, cube.data.root_ang_vel_w)
    return torch.cat([lin, ang], dim=-1)


def cube_pose_vision(
    env: "ManagerBasedRLEnv",
    noise_std_pos: float = 0.005,
    noise_std_axis: float = 0.02,
    update_period: float = 0.0,
) -> torch.Tensor:
    """ACTOR (vision-analog, gr4): cube pose in the palm frame, shaped like the
    real perception feed. VisualModule publishes the cube as a TF (position +
    orientation from measured axes: main_axis = 92mm side, table_normal) —
    deploy re-expresses that into the palm frame, so training consumes the
    same 9 numbers: position (3) + the cube's X and Z unit axes (6). The 6D
    axis form matches what vision actually measures and has no quaternion
    double-cover discontinuity. Bounded: axes are re-normalized unit vectors;
    position is clipped at the ObsTerm (every obs bounded — the gr3 law).
    Gaussian noise per step models vision jitter."""
    cube: RigidObject = env.scene["cube"]
    p_pos, p_quat = _palm_pose(env)
    pos = math_utils.quat_apply_inverse(p_quat, cube.data.root_pos_w - p_pos)
    rel_quat = math_utils.quat_mul(math_utils.quat_inv(p_quat), cube.data.root_quat_w)
    rot = math_utils.matrix_from_quat(rel_quat)
    x_axis, z_axis = rot[:, :, 0], rot[:, :, 2]
    if noise_std_pos > 0:
        pos = pos + torch.randn_like(pos) * noise_std_pos
    if noise_std_axis > 0:
        x_axis = torch.nn.functional.normalize(x_axis + torch.randn_like(x_axis) * noise_std_axis, dim=-1)
        z_axis = torch.nn.functional.normalize(z_axis + torch.randn_like(z_axis) * noise_std_axis, dim=-1)
    fresh = torch.cat([pos, x_axis, z_axis], dim=-1)
    if update_period <= 0.0:
        return fresh
    # gr6b_pose1hz: LATCHED vision (operator realism spec) — the real
    # perception chain publishes ~1 Hz, not per-step. Hold the last sample
    # between updates; episode step 0 always refreshes (post-reset).
    if not hasattr(env, "_cube_pose_latch"):
        env._cube_pose_latch = fresh.clone()
    steps = max(int(round(update_period / env.step_dt)), 1)
    refresh = (env.episode_length_buf % steps) == 0
    env._cube_pose_latch[refresh] = fresh[refresh]
    return env._cube_pose_latch


def support_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """PRIVILEGED: [retracted?, time-to-retract (clamped)] — lets the critic
    anticipate the value cliff at support removal."""
    _buffers(env)
    t = env.episode_length_buf.float() * env.step_dt
    ttr = (env.grasp_retract_t - t).clamp(-1.0, 5.0)
    return torch.stack([env.grasp_retracted.float(), ttr], dim=-1)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
def reset_hand_default(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the hand to its default state (root pose + open-finger joints).
    Without this the hand carries the previous episode's grip into the next
    one — the cube respawns inside a closed fist and the solver punches it out
    (operator: "I want it gone", 2026-08-03). Declared BEFORE reset_scene so
    the platform/cube placement reads the restored palm."""
    robot: Articulation = env.scene[asset_cfg.name]
    default_root = robot.data.default_root_state[env_ids].clone()
    default_root[:, :3] += env.scene.env_origins[env_ids]
    robot.write_root_pose_to_sim(default_root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[env_ids].clone(),
        robot.data.default_joint_vel[env_ids].clone(),
        env_ids=env_ids,
    )


def reset_grasp_scene(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    palm_xy: tuple = (0.0, 0.12),
    xy_placement_error: float = 0.03,
    gap_range: tuple = (0.02, 0.08),
    cube_height: float = 0.055,
    object_xy_offset: tuple = (0.0, 0.0),
    object_offset_wrist_frame: tuple | None = None,
    palm_track: bool = False,
    platform_thickness: float = 0.02,
    retract_time_range: tuple = (3.0, 5.0),
    approach_drop_range: tuple = (0.0, 0.0),
    approach_lateral: float = 0.0,
    approach_time_range: tuple = (0.6, 1.5),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset: place the kinematic platform so the palm-to-cube-top gap is
    U[gap_range]; drop the cube on it with random yaw and xy placement error under the
    palm; sample the support-retract time."""
    _buffers(env)
    n = len(env_ids)
    dev = env.device
    platform: RigidObject = env.scene["platform"]
    cube: RigidObject = env.scene["cube"]
    origins = env.scene.env_origins[env_ids]

    palm_pos, _ = _palm_pose(env)
    palm_z = palm_pos[env_ids, 2]

    gap = gap_range[0] + torch.rand(n, device=dev) * (gap_range[1] - gap_range[0])
    plat_top = (palm_z - gap - cube_height).clamp(min=0.05)  # first-reset stale-pose guard
    plat_pose = torch.zeros(n, 7, device=dev)
    if palm_track:
        # gr8: arm_park_error swings the real palm +-5 cm; a fixed palm_xy left
        # wide rim objects clipping the fingers at spawn. Follow the LIVE palm.
        plat_pose[:, 0] = palm_pos[env_ids, 0]
        plat_pose[:, 1] = palm_pos[env_ids, 1]
    else:
        plat_pose[:, 0] = origins[:, 0] + palm_xy[0]
        plat_pose[:, 1] = origins[:, 1] + palm_xy[1]
    plat_pose[:, 2] = plat_top - platform_thickness / 2
    plat_pose[:, 3] = 1.0

    # gr3a approach phase: spawn the support BELOW/BESIDE its final pose and let
    # approach_support() glide it in — relativity's moving palm. drop==0 -> off.
    if not hasattr(env, "grasp_plat_final"):
        env.grasp_plat_final = torch.zeros(env.num_envs, 3, device=dev)
        env.grasp_plat_start = torch.zeros(env.num_envs, 3, device=dev)
        env.grasp_app_T = torch.zeros(env.num_envs, device=dev)
    env.grasp_plat_final[env_ids] = plat_pose[:, :3]
    drop = approach_drop_range[0] + torch.rand(n, device=dev) * (
        approach_drop_range[1] - approach_drop_range[0]
    )
    app_on = drop > 1e-6
    start = plat_pose[:, :3].clone()
    start[:, 2] -= drop
    ang = torch.rand(n, device=dev) * 2 * torch.pi
    start[:, 0] += approach_lateral * torch.cos(ang) * app_on.float()
    start[:, 1] += approach_lateral * torch.sin(ang) * app_on.float()
    env.grasp_plat_start[env_ids] = start
    env.grasp_app_T[env_ids] = torch.where(
        app_on,
        approach_time_range[0]
        + torch.rand(n, device=dev) * (approach_time_range[1] - approach_time_range[0]),
        torch.zeros(n, device=dev),
    )
    plat_pose[:, :3] = start          # spawn at the approach start
    plat_top = start[:, 2] + platform_thickness / 2  # cube spawns ON the start pose
    platform.write_root_pose_to_sim(plat_pose, env_ids=env_ids)

    # object_xy_offset (gr8): displace the OBJECT (not the platform) from the
    # palm point — a wide rim object (⌀18 tube/ring) must present its WALL
    # under the palm, else the hand spawns over the open mouth.
    # object_offset_wrist_frame: same idea but rigid in the WRIST_YAW frame, so
    # it survives arm_park_error (which both translates AND rotates the hand).
    cube_pose = torch.zeros(n, 7, device=dev)
    if object_offset_wrist_frame is not None:
        w_pos, w_quat = _wrist_pose(env)
        off_local = torch.tensor(object_offset_wrist_frame, device=dev, dtype=torch.float32)
        off_w = math_utils.quat_apply(w_quat[env_ids], off_local.expand(n, 3))
        base_x = w_pos[env_ids, 0] + off_w[:, 0]
        base_y = w_pos[env_ids, 1] + off_w[:, 1]
    else:
        base_x = plat_pose[:, 0] + object_xy_offset[0]
        base_y = plat_pose[:, 1] + object_xy_offset[1]
    cube_pose[:, 0] = base_x + (torch.rand(n, device=dev) * 2 - 1) * xy_placement_error
    cube_pose[:, 1] = base_y + (torch.rand(n, device=dev) * 2 - 1) * xy_placement_error
    cube_pose[:, 2] = plat_top + cube_height / 2 + 0.002
    yaw = (torch.rand(n, device=dev) * 2 - 1) * torch.pi
    cube_pose[:, 3] = torch.cos(yaw / 2)
    cube_pose[:, 6] = torch.sin(yaw / 2)
    cube.write_root_pose_to_sim(cube_pose, env_ids=env_ids)
    cube.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    # retract clock starts AFTER the approach lands (T_app=0 when approach off)
    env.grasp_retract_t[env_ids] = env.grasp_app_T[env_ids] + retract_time_range[0] + torch.rand(
        n, device=dev
    ) * (retract_time_range[1] - retract_time_range[0])
    env.grasp_retracted[env_ids] = False


def retract_support(env: "ManagerBasedRLEnv", env_ids: torch.Tensor):
    """Interval event (every control step): once episode time passes the
    sampled retract time, teleport the platform far below — support gone,
    the grasp alone carries the cube."""
    _buffers(env)
    t = env.episode_length_buf.float() * env.step_dt
    due = (t >= env.grasp_retract_t) & (~env.grasp_retracted)
    ids = torch.nonzero(due).squeeze(-1)
    if len(ids) == 0:
        return
    platform: RigidObject = env.scene["platform"]
    pose = platform.data.root_pose_w[ids].clone()
    pose[:, 2] = -5.0
    platform.write_root_pose_to_sim(pose, env_ids=ids)
    env.grasp_retracted[ids] = True


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
def _closure_touching(env: "ManagerBasedRLEnv", force_thr: float = 0.5) -> torch.Tensor:
    """gr6c 'multiple fingers' gate: 1.0 only when a THUMB pad AND at least one
    OPPOSING FINGER pad are both loaded (palm pad ignored entirely). The
    gr6b palm-press autopsy: `touching = pad_force_sum > 1` counts the palm,
    so pressing the cube with the palm satisfied every touching gate — the
    0.3 any-contact tier of pad_arrangement paid while its 0.7 closure tier
    never fired (HUD-verified, 2026-08-19). This is the closure tier's
    definition reused as a GATE."""
    f = _pad_force_mags(env)
    thumb_m, finger_m = _pad_masks(env)
    loaded = f > force_thr
    return (loaded[:, thumb_m].any(dim=-1) & loaded[:, finger_m].any(dim=-1)).float()


def hold_cube_bonus(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.06,
    gate_mode: str = "any",
    ramp_lo: float = 0.01,
    ramp_hi: float = 0.05,
) -> torch.Tensor:
    """THE dish. After support removal: exp(-(d/sigma)^2) on cube-to-palm
    distance. Bounded (the anchor2 lesson: bonuses, not unbounded penalties)."""
    _buffers(env)
    cube: RigidObject = env.scene["cube"]
    p_pos, _ = _palm_pose(env)
    d = (cube.data.root_pos_w - p_pos).norm(dim=-1)
    # gr2: the old gate was 0.25 + 0.75*retracted, i.e. a flat 25% payment for the
    # cube merely being NEAR the palm before retract — which the platform provides
    # for free. Measured: doing nothing earned +0.10/episode from this term while
    # never touching the cube (pad_force_sum = 0.000 N). Now the pre-retract share
    # is small AND requires actual pad contact, so proximity alone pays ~nothing
    # and the prize lives where it belongs: after the support is gone.
    # gr6c: gate_mode "closure" swaps the palm-inclusive any-pad touching for
    # thumb+opposing-finger closure (the palm-press loophole); ramp_lo 0.025
    # kills the palm-TILT variant (edge-tilting a 5 cm cube raises its center
    # ~1 cm — exactly the old lo=0.01 threshold, so tilts collected ramp).
    if gate_mode == "closure":
        touching = _closure_touching(env)
    else:
        touching = (_pad_force_mags(env).sum(dim=-1) > 1.0).float()
    # gr6b GATE FIX (Bible rule 25 corollary — port the SEMANTICS, not the
    # code): "the prize lives after the support is gone" meant grasp_retracted
    # in gr5; on the PERMANENT table the retract event no longer exists, the
    # flag never fired, and the +40 dish sat capped at 5% for the whole gr6
    # generation (measured 0.2-0.5/s — the operator's wandb read caught it).
    # The permanent-table translation: the hand supports the cube = the cube
    # is OFF the table. RAMPED, not stepped ("a promise is not a gradient"):
    # the 95% share scales 0 -> 1 over cube height [ramp_lo, ramp_hi] above
    # the platform top, so lifting pays from the first centimeter.
    gate = 0.05 * touching + 0.95 * touching * _lift_ramp(env, ramp_lo, ramp_hi)
    return torch.exp(-((d / sigma) ** 2)) * gate


def pad_arrangement_bonus(
    env: "ManagerBasedRLEnv", force_thr: float = 0.5, closure_mode: str = "any"
) -> torch.Tensor:
    """Dense pre-grasp shaping: 0.3 for any pad contact, +0.7 for FORCE
    CLOSURE (a thumb pad AND an opposing finger pad both loaded) — rewards
    grasps, not scoops.

    closure_mode:
      "any"   — gr2b behaviour: closure is binary, thumb + at least ONE
                opposing finger (this is how the thumb+pinky grip got full pay)
      "count" — gr3d: closure scales with HOW MANY of the four opposing
                fingers are loaded (thumb_loaded * n_fingers/4): thumb+1 pays
                0.3+0.175, thumb+4 pays the full 1.0 — load sharing is paid
                directly, wrap grasps dominate pinches, no grip is prescribed."""
    f = _pad_force_mags(env)
    thumb_m, finger_m = _pad_masks(env)
    loaded = f > force_thr
    any_c = loaded.any(dim=-1).float()
    thumb_any = loaded[:, thumb_m].any(dim=-1).float()
    if closure_mode == "count":
        names = env.scene.sensors["pad_sensor"].body_names
        if not hasattr(env, "grasp_finger_group"):
            import re
            groups = torch.full((len(names),), -1, dtype=torch.long, device=env.device)
            for gi, key in enumerate(("index", "middle", "ring", "little")):
                for bi, bn in enumerate(names):
                    if key in bn:
                        groups[bi] = gi
            env.grasp_finger_group = groups
        per_finger = torch.zeros(f.shape[0], 4, device=env.device, dtype=torch.bool)
        for gi in range(4):
            m = env.grasp_finger_group == gi
            if m.any():
                per_finger[:, gi] = loaded[:, m].any(dim=-1)
        closure = thumb_any * per_finger.float().sum(dim=-1) / 4.0
    else:
        closure = thumb_any * loaded[:, finger_m].any(dim=-1).float()
    return 0.3 * any_c + 0.7 * closure


def crush_penalty(
    env: "ManagerBasedRLEnv", total_thr: float = 30.0, max_excess: float = 100.0
) -> torch.Tensor:
    """Hinged penalty on total pad force above threshold — the anti-burnout
    pattern (hinged cap, not blanket tau^2): squeezing hard enough to hold is
    free, crushing is not.

    gr2b: now BOUNDED (max_excess). The unbounded form was the direct cause of
    gr2a's blowup: pad force is in newtons with no ceiling, so a finger slam
    produced an arbitrarily large penalty, the value target followed it, and the
    run went -2.7e13 -> -3.5e25 within ~50 iterations. Same failure family as
    anchor1's unbounded -30*d^2 attractor; same fix (bound it). Clipping costs
    nothing behaviourally — anything past +100 N of excess is already a hard
    crush, and the gradient below the cap is unchanged."""
    total = _pad_force_mags(env).sum(dim=-1)
    return (total - total_thr).clamp(min=0.0, max=max_excess)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
def cube_dropped(env: "ManagerBasedRLEnv", z_threshold: float = 0.08) -> torch.Tensor:
    """Cube fell to (near) the floor — after retract this is a failed grasp;
    before retract it fell off the platform. Either way the episode is over."""
    cube: RigidObject = env.scene["cube"]
    return (cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]) < z_threshold


# ---------------------------------------------------------------------------
# gr3 kit
# ---------------------------------------------------------------------------
def action_rate_clamped(env: "ManagerBasedRLEnv", max_sq: float = 25.0) -> torch.Tensor:
    """gr3: BOUNDED action-rate penalty. The unbounded base_mdp.action_rate_l2
    killed gr2_grasp_b at it 720: rising exploration std grew raw-action jitter,
    the quadratic hit -3.1e6/episode, the value target followed, NaN std — the
    third unbounded-term explosion of the gr line (crush gr2a, action_rate
    gr2b). Sum of squared deltas, clamped per env per step. 6 actions in
    [-1,1]: normal operation sums <1; the 25.0 cap only engages in the flail
    regime the clamp exists to defuse."""
    d = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(d), dim=-1).clamp(max=max_sq)


def _driver_ids(env: "ManagerBasedRLEnv"):
    if not hasattr(env, "grasp_driver_ids"):
        asset: Articulation = env.scene["robot"]
        ids, _ = asset.find_joints(DRIVER_JOINTS, preserve_order=True)
        env.grasp_driver_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    return env.grasp_driver_ids


def finger_vel_reversal(env: "ManagerBasedRLEnv", max_sq: float = 4.0) -> torch.Tensor:
    """gr4 (operator design, 2026-08-05): penalize DIRECTION REVERSALS of the
    finger driver joints — pay relu(-dq_t * dq_{t-1}) per joint. Punishes
    exactly the free-finger dither (flip-flopping velocity) while a fast
    COMMITTED close (monotone dq) costs nothing — which a plain joint_acc
    penalty would also tax. Clamped (every term bounded — the gr law); zeroed
    on the reset step (prev dq belongs to the previous episode)."""
    ids = _driver_ids(env)
    dq = env.scene["robot"].data.joint_vel[:, ids]
    if not hasattr(env, "grasp_prev_dq"):
        env.grasp_prev_dq = torch.zeros_like(dq)
    rev = torch.relu(-(dq * env.grasp_prev_dq)).sum(dim=-1).clamp(max=max_sq)
    env.grasp_prev_dq = dq.clone()
    rev = torch.where(env.episode_length_buf <= 1, torch.zeros_like(rev), rev)
    return rev


def finger_jacc_clamped(env: "ManagerBasedRLEnv", scale: float = 1e-4,
                        max_val: float = 10.0) -> torch.Tensor:
    """gr4: bounded joint-acceleration penalty on the finger drivers — the
    smoothness bracket's other rung (vs finger_vel_reversal). scale*sum(acc^2),
    clamped per env per step."""
    ids = _driver_ids(env)
    acc = env.scene["robot"].data.joint_acc[:, ids]
    return (scale * torch.sum(torch.square(acc), dim=-1)).clamp(max=max_val)


def finger_contact_count(env: "ManagerBasedRLEnv", force_thr: float = 0.5) -> torch.Tensor:
    """gr4 (operator design, 2026-08-05): 'each finger touching the cube gives
    a reward — the more the merrier'. Pays (thumb + 4 opposing fingers in
    contact)/5 per tick — a DENSE version of the count-closure idea: every
    additional finger recruited pays immediately, incl. the parked middle
    finger. Bounded <= 1 by construction."""
    f = _pad_force_mags(env)
    thumb_m, _ = _pad_masks(env)
    loaded = f > force_thr
    names = env.scene.sensors["pad_sensor"].body_names
    if not hasattr(env, "grasp_finger_group"):
        groups = torch.full((len(names),), -1, dtype=torch.long, device=env.device)
        for gi, key in enumerate(("index", "middle", "ring", "little")):
            for bi, bn in enumerate(names):
                if key in bn:
                    groups[bi] = gi
        env.grasp_finger_group = groups
    n = loaded[:, thumb_m].any(dim=-1).float()
    for gi in range(4):
        m = env.grasp_finger_group == gi
        if m.any():
            n = n + loaded[:, m].any(dim=-1).float()
    return n / 5.0


def approach_support(env: "ManagerBasedRLEnv", env_ids: torch.Tensor):
    """gr3a: MOVING-PALM curriculum, implemented by relativity. The policy is
    blind to world pose (tactile + drivers only), so moving the SUPPORT+CUBE
    toward the palm is observationally identical to the palm approaching the
    cube — with zero articulation surgery. Each control step, envs still in
    their approach window get the platform smoothstep-lerped from its spawned
    start pose (below/beside the final pose) to the final pose; the cube rides
    on top (kinematic platform, gentle speeds, friction carries it). No-op for
    envs with approach disabled (T_app == 0) or already retracted."""
    _buffers(env)
    if not hasattr(env, "grasp_app_T"):
        return
    t = env.episode_length_buf.float() * env.step_dt
    active = (env.grasp_app_T > 0.0) & (~env.grasp_retracted) & (t < env.grasp_app_T + 0.1)
    ids = torch.nonzero(active).squeeze(-1)
    if len(ids) == 0:
        return
    platform: RigidObject = env.scene["platform"]
    f = (t[ids] / env.grasp_app_T[ids]).clamp(0.0, 1.0)
    s = f * f * (3.0 - 2.0 * f)                      # smoothstep
    pos = env.grasp_plat_start[ids] + (env.grasp_plat_final[ids] - env.grasp_plat_start[ids]) * s.unsqueeze(-1)
    pose = torch.zeros(len(ids), 7, device=env.device)
    pose[:, :3] = pos
    pose[:, 3] = 1.0
    platform.write_root_pose_to_sim(pose, env_ids=ids)
    platform.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


def pad_contacts_noisy(
    env: "ManagerBasedRLEnv",
    thr_range: tuple = (0.3, 1.5),
    dropout_p: float = 0.05,
) -> torch.Tensor:
    """gr3c: BINARIZED tactile with per-episode sensor DR — the deploy-contract
    answer to 'the FTP tactile has no sim model'. Instead of asking the policy
    to trust sim force MAGNITUDES (which real taxel pooling will not reproduce),
    give it contact BOOLEANS behind a randomized threshold: per episode, each of
    the 17 pads draws its own threshold U[thr_range] and is dead (stuck at 0)
    with p=dropout_p. Real-side mapping becomes 'pooled taxel value > calib
    threshold', which survives any monotone calibration. 17-dim, same slot
    count as pad_forces_log."""
    f = _pad_force_mags(env)
    n, p = f.shape
    if not hasattr(env, "grasp_tact_thr"):
        env.grasp_tact_thr = torch.full((n, p), 0.5, device=env.device)
        env.grasp_tact_alive = torch.ones(n, p, device=env.device)
    fresh = env.episode_length_buf == 0
    if fresh.any():
        ids = torch.nonzero(fresh).squeeze(-1)
        env.grasp_tact_thr[ids] = thr_range[0] + torch.rand(len(ids), p, device=env.device) * (
            thr_range[1] - thr_range[0]
        )
        env.grasp_tact_alive[ids] = (torch.rand(len(ids), p, device=env.device) > dropout_p).float()
    return (f > env.grasp_tact_thr).float() * env.grasp_tact_alive


# ---------------------------------------------------------------------------
# gr3b: residual wrist — 6-DoF pose offset around the spawn pose
# ---------------------------------------------------------------------------
class WristPoseAction(ActionTerm):
    """gr3b: the hand root becomes a slow 6-DoF residual (floating base, gravity
    off, pose written kinematically every control step, root velocity zeroed).
    Actions in [-1,1]^6 integrate into a pose OFFSET from the spawn pose:
    per-step slew pos_step/rot_step, hard box pos_limit/rot_limit — the policy
    does last-centimetre alignment, it cannot fly away. Deploy analog: the
    offset adds to the IK-resolved wrist target BEFORE the colleague's resolver
    runs (ActionModule integration, to be agreed before any hardware use)."""

    cfg: "WristPoseActionCfg"

    def __init__(self, cfg: "WristPoseActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._raw = torch.zeros(self.num_envs, 6, device=self.device)
        self._off_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._off_rpy = torch.zeros(self.num_envs, 3, device=self.device)
        d = self._asset.data.default_root_state
        self._home_pos = d[:, :3] + env.scene.env_origins
        self._home_quat = d[:, 3:7].clone()
        # export_deploy_cfg walks _joint_ids on every action term; this term
        # drives the ROOT, not joints — empty list is the honest contract. Its
        # deploy path (offset onto the IK wrist target) lives outside the
        # joint-command pipeline entirely.
        self._joint_ids = []

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.cat([self._off_pos, self._off_rpy], dim=-1)

    def process_actions(self, actions: torch.Tensor):
        self._raw = actions.clamp(-1.0, 1.0)
        self._off_pos = (self._off_pos + self._raw[:, :3] * self.cfg.pos_step).clamp(
            -self.cfg.pos_limit, self.cfg.pos_limit
        )
        self._off_rpy = (self._off_rpy + self._raw[:, 3:] * self.cfg.rot_step).clamp(
            -self.cfg.rot_limit, self.cfg.rot_limit
        )

    def apply_actions(self):
        dq = math_utils.quat_from_euler_xyz(
            self._off_rpy[:, 0], self._off_rpy[:, 1], self._off_rpy[:, 2]
        )
        quat = math_utils.quat_mul(dq, self._home_quat)
        pose = torch.cat([self._home_pos + self._off_pos, quat], dim=-1)
        self._asset.write_root_pose_to_sim(pose)
        self._asset.write_root_velocity_to_sim(
            torch.zeros(self.num_envs, 6, device=self.device)
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            self._off_pos.zero_(); self._off_rpy.zero_()
        else:
            self._off_pos[env_ids] = 0.0; self._off_rpy[env_ids] = 0.0


@configclass
class WristPoseActionCfg(ActionTermCfg):
    class_type: type = WristPoseAction
    asset_name: str = "robot"
    # export_deploy_cfg introspection contract (same as CoupledFingerActionCfg):
    # raw actions are [-1,1] deltas; the term itself owns the step/limit mapping
    scale: float = 1.0
    clip = None
    pos_step: float = 0.004    # m per control step at full action (0.2 m/s)
    rot_step: float = 0.02     # rad per control step (1 rad/s)
    pos_limit: float = 0.06    # m box around spawn
    rot_limit: float = 0.30    # rad box around spawn


def wrist_pose_offset(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """gr3b obs: the wrist's pose offset from its spawn pose (pos/limit,
    rpy/limit -> roughly [-1,1]^6) so the actor knows where its residual sits
    inside the allowed box. Computed from sim state, not the action buffer."""
    asset: Articulation = env.scene["robot"]
    d = asset.data.default_root_state
    home_pos = d[:, :3] + env.scene.env_origins
    dp = (asset.data.root_pos_w - home_pos) / 0.06
    qrel = math_utils.quat_mul(asset.data.root_quat_w, math_utils.quat_conjugate(d[:, 3:7]))
    r, p, y = math_utils.euler_xyz_from_quat(qrel)
    drpy = torch.stack([r, p, y], dim=-1) / 0.30
    return torch.cat([dp, drpy], dim=-1)


# ---------------------------------------------------------------------------
# gr5b: arm-in-the-loop (2026-08-06)
# ---------------------------------------------------------------------------
def reset_arm_park_error(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    joint_names: list,
    noise_rad: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Deploy-analog PARKING ERROR: after reset_hand_default restores the
    default pose, add uniform joint noise to the ARM joints — the resolver
    parked with error, and the policy's arm channel can now CORRECT it
    (the gr5 aligndr/offsetdr tail becomes solvable). Bounded by noise_rad."""
    robot: Articulation = env.scene[asset_cfg.name]
    key = "_gr5b_arm_ids_" + str(len(joint_names))
    if not hasattr(env, key):
        ids, _ = robot.find_joints(joint_names)
        setattr(env, key, torch.tensor(ids, device=env.device, dtype=torch.long))
    jids = getattr(env, key)
    n = len(env_ids)
    noise = (torch.rand(n, len(jids), device=env.device) * 2.0 - 1.0) * noise_rad
    jp = robot.data.joint_pos[env_ids].clone()
    jp[:, jids] += noise
    jv = robot.data.joint_vel[env_ids].clone()
    jv[:, jids] = 0.0
    robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)


def table_contact_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("hand_contact"),
    force_thr: float = 1.0,
    max_val: float = 10.0,
) -> torch.Tensor:
    """Penalize HAND-STRUCTURE contact with the table slab (thumb strikes —
    the real reason production pitches the wrist -24 deg). Filtered contact
    forces vs the platform only, hinged above force_thr, CLAMPED (gr law)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fm = sensor.data.force_matrix_w  # (N, bodies, filters, 3)
    if fm is None:
        return torch.zeros(env.num_envs, device=env.device)
    mag = torch.norm(fm, dim=-1).sum(dim=(-2, -1))
    # NaN/Inf guard (dp5b) — clamp() passes NaN through untouched.
    mag = torch.nan_to_num(mag, nan=0.0, posinf=max_val + force_thr, neginf=0.0)
    return (mag - force_thr).clamp(min=0.0, max=max_val)


def joint_vel_reversal(
    env: "ManagerBasedRLEnv",
    joint_names: list,
    max_sq: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """finger_vel_reversal generalized to an arbitrary joint set (gr5b arm
    smoothness): penalize velocity direction flips, clamped, reset-safe."""
    robot: Articulation = env.scene[asset_cfg.name]
    key = "_jvr_ids_" + str(hash(tuple(joint_names)) % 100000)
    if not hasattr(env, key):
        ids, _ = robot.find_joints(joint_names)
        setattr(env, key, torch.tensor(ids, device=env.device, dtype=torch.long))
    jids = getattr(env, key)
    dq = robot.data.joint_vel[:, jids]
    pkey = key + "_prev"
    if not hasattr(env, pkey):
        setattr(env, pkey, torch.zeros_like(dq))
    prev = getattr(env, pkey)
    rev = torch.relu(-(dq * prev)).sum(dim=-1).clamp(max=max_sq)
    setattr(env, pkey, dq.clone())
    return rev


def wobble_root(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    pos_range: float = 0.02,
    rot_range: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """gr5b_armwobble: TORSO-MOUNT WOBBLE DR — the balance policy's residual
    sway, emulated by re-jittering the fixed root pose (reset mode: per-episode
    mount error; interval mode: slow sway). Small by construction."""
    robot: Articulation = env.scene[asset_cfg.name]
    n = len(env_ids)
    default = robot.data.default_root_state[env_ids].clone()
    default[:, :3] += env.scene.env_origins[env_ids]
    pr = torch.tensor(pos_range if isinstance(pos_range, (tuple, list)) else (pos_range,) * 3,
                      device=env.device, dtype=default.dtype)
    default[:, :3] += (torch.rand(n, 3, device=env.device) * 2 - 1) * pr
    dr = (torch.rand(n, 3, device=env.device) * 2 - 1) * rot_range
    q = default[:, 3:7]
    half = dr * 0.5
    dq = torch.cat([torch.ones(n, 1, device=env.device), half], dim=-1)
    dq = dq / dq.norm(dim=-1, keepdim=True)
    w1, x1, y1, z1 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w2, x2, y2, z2 = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]
    default[:, 3] = w1*w2 - x1*x2 - y1*y2 - z1*z2
    default[:, 4] = w1*x2 + x1*w2 + y1*z2 - z1*y2
    default[:, 5] = w1*y2 - x1*z2 + y1*w2 + z1*x2
    default[:, 6] = w1*z2 + x1*y2 - y1*x2 + z1*w2
    robot.write_root_pose_to_sim(default[:, :7], env_ids=env_ids)


def capture_cube_ref_pose(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
):
    """gr5c: snapshot the cube's PALM-FRAME pose at reset — i.e. the pose the
    VisualModule reported before the grasp. `cube_pose_hold_bonus` pays for
    still being in that pose later. Declare AFTER reset_scene (it must read the
    placed cube)."""
    cube: RigidObject = env.scene["cube"]
    p_pos, p_quat = _palm_pose(env)
    pos_b = math_utils.quat_apply_inverse(p_quat, cube.data.root_pos_w - p_pos)
    quat_b = math_utils.quat_mul(math_utils.quat_inv(p_quat), cube.data.root_quat_w)
    if not hasattr(env, "cube_ref_pos_b"):
        env.cube_ref_pos_b = torch.zeros_like(pos_b)
        env.cube_ref_quat_b = torch.zeros_like(quat_b)
        env.cube_ref_quat_b[:, 0] = 1.0
    env.cube_ref_pos_b[env_ids] = pos_b[env_ids]
    env.cube_ref_quat_b[env_ids] = quat_b[env_ids]


def cube_pose_hold_bonus(
    env: "ManagerBasedRLEnv",
    sigma_pos: float = 0.04,
    sigma_rot: float = 0.35,
    force_thr: float = 0.5,
) -> torch.Tensor:
    """gr5c (operator 2026-08-10): keep the cube in the pose it was TAKEN in.

    Rationale: in sim the support platform retracts, so the policy is free to
    re-orient the cube down into space the REAL TABLE still occupies — it
    "turns it into a more favourable position that would be under the table".
    On the robot that is a collision. This term pays for holding the cube at
    the palm-frame pose the vision stack reported (captured by
    capture_cube_ref_pose at reset).

    Shape follows the project's design grammar — a bounded kernel-near INCOME
    (a goal to hold), not a penalty (penalties are for things that must never
    happen: crush, table strike, limit riding):
        exp(-|dpos|^2/sigma_pos^2) * exp(-(angle_err/sigma_rot)^2),  in [0, 1]
    CONTACT-GATED: pays nothing before the fingers are actually loaded, so it
    cannot reward a policy for hovering next to an untouched cube.
    NB it biases, never vetoes — at a modest weight it cannot fight the arm/IK
    authority for the pose the resolver wants (operator's IK-conflict question).
    """
    if not hasattr(env, "cube_ref_pos_b"):
        return torch.zeros(env.num_envs, device=env.device)
    cube: RigidObject = env.scene["cube"]
    p_pos, p_quat = _palm_pose(env)
    pos_b = math_utils.quat_apply_inverse(p_quat, cube.data.root_pos_w - p_pos)
    quat_b = math_utils.quat_mul(math_utils.quat_inv(p_quat), cube.data.root_quat_w)
    d2 = torch.sum((pos_b - env.cube_ref_pos_b) ** 2, dim=-1)
    # geodesic angle between current and reference orientation, sign-safe
    dot = torch.sum(quat_b * env.cube_ref_quat_b, dim=-1).abs().clamp(max=1.0)
    ang = 2.0 * torch.acos(dot)
    bonus = torch.exp(-d2 / (sigma_pos ** 2)) * torch.exp(-(ang / sigma_rot) ** 2)
    # contact gate: any pad loaded
    forces = _pad_force_mags(env)
    holding = (forces.max(dim=-1).values > force_thr).float()
    return bonus * holding


def quiet_hold_bonus(
    env: "ManagerBasedRLEnv",
    joint_names: list | None = None,
    sigma: float = 3.0,
    force_thr: float = 0.5,
) -> torch.Tensor:
    """gr5c: INCOME for a CALM arm while actually holding the cube.

    The gr5b verdict was "too chaotic — brash and quick" (operator 2026-08-11).
    Measured on gr5b_arm's deterministic policy over 7 arm joints:
        sum(qdot^2): median 0.041, p75 0.489, p90 38.6, MAX 2408 (~18 rad/s/joint)
    i.e. the mean is 247x the median — the arm is calm most of the time and
    violently bursty in the tail.

    WHY THIS SHAPE, and not a velocity penalty:
      * a per-step velocity cost is a COST — a robot doing the job must move, so
        it can never pay ~zero (the project's constraint-vs-cost test).
      * `arm_smooth` (joint_vel_reversal) structurally CANNOT catch this: a
        single fast sweep is not a reversal, which is why it only ever paid
        -0.03 while the arm was reaching 18 rad/s.
      * L1 on the velocity NORM, not L2 on the sum of squares: with
        exp(-sum(qdot^2)/sigma^2) the p90 burst reads exp(-38.6/9) ~ 0 and has
        NO gradient, so violent states would be un-improvable ("pay for the
        journey"). On the norm with sigma 3.0 the measured quantiles read
        median 0.94 / p75 0.79 / p90 0.13 — graded everywhere.
      * CONTACT-GATED, so it cannot pay a policy for holding still next to an
        untouched cube. The gate is already reachable (hold_cube collects ~38%
        of max today), which is what `cube_hold` at sigma_pos 0.04 was not — it
        collected 4% and taught us nothing.
    """
    asset: Articulation = env.scene["robot"]
    if joint_names is None:
        joint_names = ARM7_JOINTS
    # regex-tolerant (side-agnostic names): find_joints, cached per name set
    key = "_gr_jn_ids_" + str(hash(tuple(joint_names)) % 100000)
    if not hasattr(env, key):
        ids, _ = asset.find_joints(joint_names, preserve_order=True)
        setattr(env, key, ids)
    idx = getattr(env, key)
    speed = torch.linalg.norm(asset.data.joint_vel[:, idx], dim=-1)
    speed = torch.nan_to_num(speed, nan=0.0, posinf=1e3, neginf=0.0)
    bonus = torch.exp(-speed / sigma)
    forces = _pad_force_mags(env)
    holding = (forces.max(dim=-1).values > force_thr).float()
    return bonus * holding


def park_keep_bonus(
    env: "ManagerBasedRLEnv",
    joint_names: list | None = None,
    sigma: float = 1.7,
    ref: str = "default",
) -> torch.Tensor:
    """gr5c: INCOME for keeping the arm near its PARK pose (the IK-solved
    production grasp pose that is `default_joint_pos` for these joints).

    The gr5b verdict was "it flings its arm towards the robot's back instead of
    leaving the hand/arm in front" (operator 2026-08-11). Nothing in the gr5b
    ledger rewarded arm POSE at all — `reset_arm_park_error` is only a reset
    event — so a pose behind the robot was free as long as the cube stayed in
    the palm.

    This is the POSITIVE form of "don't fling it behind you": an attractor at
    the pose the resolver wants, rather than a tax on being elsewhere. It
    mirrors deploy, where ERNEST owns the journey and the policy owns the last
    five centimetres.

    sigma 1.7 from measurement, NOT guessed: |q - q_park|^2 over the 7 arm
    joints reads median 2.95 (|dq| = 1.72 rad) on gr5b_arm today, so the kernel
    collects exp(-1) = 37% at the CURRENT pose — reachable, with gradient in
    both directions. NOT contact-gated: the arm should stay home whether or not
    it is holding anything.
    """
    asset: Articulation = env.scene["robot"]
    if joint_names is None:
        joint_names = ARM7_JOINTS
    # regex-tolerant (side-agnostic names): find_joints, cached per name set
    key = "_gr_jn_ids_" + str(hash(tuple(joint_names)) % 100000)
    if not hasattr(env, key):
        ids, _ = asset.find_joints(joint_names, preserve_order=True)
        setattr(env, key, ids)
    idx = getattr(env, key)
    # gr8c ref="start": the attractor is the EPISODE-START pose (the pose the
    # resolver delivered), not the frozen default — under handover-relative
    # actions an attractor at default would pay the exact yank-home behaviour
    # the wave removes. Falls back to default until the action term stashes it.
    if ref == "start" and hasattr(env, "gr8c_start_arm_q"):
        home = env.gr8c_start_arm_q
    else:
        home = asset.data.default_joint_pos[:, idx]
    dev = asset.data.joint_pos[:, idx] - home
    d2 = torch.sum(dev * dev, dim=-1)
    d2 = torch.nan_to_num(d2, nan=0.0, posinf=1e3, neginf=0.0)
    return torch.exp(-d2 / (sigma ** 2))


# ── gr8c: HANDOVER-RELATIVE actions ──────────────────────────────────────────
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction as _JPA
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg as _JPACfg


class RelStartJointPositionAction(_JPA):
    """gr8c (operator 2026-09-01): q_target = q_AT_EPISODE_START + scale*a.

    The gr8b decode (default_pose + 0.15*a) freezes the reference at training
    time; at deploy the policy's first command yanks the arm from wherever the
    resolver parked it back toward that frozen pose (measured: a converged
    1.5 cm hover became 6 cm the moment the override started, 2026-09-01 rig).
    Here the reference is the pose the arm HOLDS when the episode begins —
    after reset_arm_park_error has randomized it — so at deploy the
    JointCommander simply latches the measured arm at override start and the
    policy works from wherever IK delivered it. Before the first reset the
    offset is the default pose (use_default_offset=True path), so a gr8b
    warmstart starts from an identical decode and adapts smoothly.
    Also stashes env.gr8c_start_arm_q for start-referenced reward terms
    (park_keep ref="start")."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not isinstance(self._offset, torch.Tensor):
            self._offset = torch.zeros_like(self._raw_actions) + float(self._offset)
        self._env_ref = env

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        q = self._asset.data.joint_pos[:, self._joint_ids]
        self._offset[ids] = q[ids]
        env = self._env_ref
        if not hasattr(env, "gr8c_start_arm_q"):
            env.gr8c_start_arm_q = q.clone()
        env.gr8c_start_arm_q[ids] = q[ids]


@configclass
class RelStartJointPositionActionCfg(_JPACfg):
    class_type: type = RelStartJointPositionAction


def capture_hand_start(env: "ManagerBasedRLEnv", env_ids):
    """gr6: store each env's PALM spawn position (world). The task's hold
    target: 'where the hand started is where the cube must be held' —
    operator redesign 2026-08-13, replacing the disappearing-table race."""
    import torch as _t
    p_pos, _ = _palm_pose(env)
    if not hasattr(env, "hand_start_pos_w"):
        env.hand_start_pos_w = p_pos.clone()
    env.hand_start_pos_w[env_ids] = p_pos[env_ids]
    # gr6b: also capture the CUBE start (the hold target xy anchor);
    # reset_cube_retry updates it again for seeded envs it moves.
    cube_pos = env.scene["cube"].data.root_pos_w
    if not hasattr(env, "cube_start_pos_w"):
        env.cube_start_pos_w = cube_pos.clone()
    env.cube_start_pos_w[env_ids] = cube_pos[env_ids]


def cube_at_start_bonus(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.05,
    force_thr: float = 0.5,
) -> torch.Tensor:
    """gr6 THE DISH: bounded kernel for holding the CUBE at the hand's spawn
    point, CONTACT-GATED (fingers loaded). The target sits ABOVE the permanent
    table, so lifting is the only way to collect and 'mushing' the cube around
    the tabletop pays nothing (operator: 'no table disappears in real life;
    it should be picked up, not dragged'). Speed is priced implicitly: every
    second not holding is income lost forever — no explicit speed bonus (the
    Bible: don't double-pay)."""
    if not hasattr(env, "hand_start_pos_w"):
        return torch.zeros(env.num_envs, device=env.device)
    cube: RigidObject = env.scene["cube"]
    d2 = torch.sum((cube.data.root_pos_w - env.hand_start_pos_w) ** 2, dim=-1)
    d2 = torch.nan_to_num(d2, nan=1e3, posinf=1e3, neginf=1e3)
    bonus = torch.exp(-d2 / (sigma ** 2))
    forces = _pad_force_mags(env)
    holding = (forces.max(dim=-1).values > force_thr).float()
    return bonus * holding


# ---------------------------------------------------------------------------
# gr6b: the corrected permanent-table economy (2026-08-18)
# ---------------------------------------------------------------------------

def _table_top_w(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-env platform top z (world). Platform slab is 0.02 thick."""
    return env.scene["platform"].data.root_pos_w[:, 2] + 0.01


def _lift_ramp(env: "ManagerBasedRLEnv", lo: float = 0.01, hi: float = 0.05) -> torch.Tensor:
    """0 -> 1 as the cube rises from lo to hi above the platform top.
    The gradient bridge between resting and held-aloft; NaN-safe (clamp of
    finite physics values; the cube cannot be NaN without root_out_of_bounds
    class guards tripping upstream in the balance line — grasp scenes have no
    push events, but the clamp itself is benign either way)."""
    cube: RigidObject = env.scene["cube"]
    h = cube.data.root_pos_w[:, 2] - _table_top_w(env)
    return ((h - lo) / (hi - lo)).clamp(0.0, 1.0)


def cube_hold_above_bonus(
    env: "ManagerBasedRLEnv",
    height: float = 0.15,
    sigma: float = 0.06,
    gate_mode: str = "any",
    ramp_lo: float = 0.01,
    ramp_hi: float = 0.05,
    height_only: bool = False,
) -> torch.Tensor:
    """gr6b: hold the cube at a FIXED HEIGHT above the table (operator design:
    'It shouldn't be start_pose height we are targeting but a fixed distance
    from table'). Target = the cube's own start xy at platform_top + height.
    Replaces cube_at_start, whose target (the IK hand-spawn point, measured
    z 1.086 vs cube 1.035) sat ~5 cm above the resting cube — resting
    collected ~30% riskless. Income = kernel x touching x lift ramp: resting
    pays ZERO from this term."""
    _buffers(env)
    cube: RigidObject = env.scene["cube"]
    if gate_mode == "closure":
        touching = _closure_touching(env)
    else:
        touching = (_pad_force_mags(env).sum(dim=-1) > 1.0).float()
    target = env.cube_start_pos_w.clone()
    target[:, 2] = _table_top_w(env) + height
    if height_only:
        # gr6f (2026-08-25): a real grasp drags/rotates the cube 5-10 cm from its
        # START xy, and the 3-D kernel at sigma 0.06 paid ~10-40% for a hold at
        # the CORRECT height (HUD: equalhold held at height, term barely fired).
        # Score the height alone; lateral drift is not this term's business.
        d = (cube.data.root_pos_w[:, 2] - target[:, 2]).abs()
    else:
        d = (cube.data.root_pos_w - target).norm(dim=-1)
    return torch.exp(-((d / sigma) ** 2)) * touching * _lift_ramp(env, ramp_lo, ramp_hi)


def cube_slide_penalty(
    env: "ManagerBasedRLEnv", ramp_lo: float = 0.01, ramp_hi: float = 0.05, max_speed: float = 1.0
) -> torch.Tensor:
    """gr7b (2026-08-26): the CLEAN-PICK term. gr6f_combo grasps and holds at the
    right height, but 'muffles the cube around and only then picks it up'
    (operator). Nothing paid for a straight pick and nothing punished shoving:
    cube_hold_above only starts above the lift ramp and the xy anchor was
    deliberately removed (height_only). Penalize the cube's horizontal speed
    WHILE IT IS STILL ON THE TABLE — x(1 - lift_ramp) — so lifting straight up
    costs nothing and a slide/shove costs |v_xy| per step. Clamped (gr law):
    a flung cube reads at most `max_speed` m/s."""
    cube: RigidObject = env.scene["cube"]
    v_xy = cube.data.root_lin_vel_w[:, :2].norm(dim=-1)
    v_xy = torch.nan_to_num(v_xy, nan=0.0, posinf=max_speed, neginf=0.0).clamp(max=max_speed)
    return v_xy * (1.0 - _lift_ramp(env, ramp_lo, ramp_hi))


def object_orientation_deviation(
    env: "ManagerBasedRLEnv",
    ramp_lo: float = 0.01,
    ramp_hi: float = 0.05,
    capture_at: float = 0.5,
    max_angle: float = 1.57,
) -> torch.Tensor:
    """gr8 (operator 2026-08-27): 'held with intention, placeable later' — the
    joint_deviation idea applied to the OBJECT. Capture the object's world
    orientation the first time the lift ramp crosses `capture_at` (= the pickup
    moment), then charge the geodesic angle from that reference x ramp while it
    stays aloft. Gated by the ramp so the pre-grasp phase is never punished;
    penalizes DRIFT WHILE HELD, not an imperfect pickup pose. The reference
    self-clears when the ramp returns to 0 (object back on the table, or the
    episode reset re-placed it), so no reset hook is needed. Clamped at
    `max_angle` (gr law: a tumbling object reads at most ~90deg per step).
    For the ring this is the difference between hooked-and-dangling and a
    grasp that could later PLACE the ring — and it prices the edge-standing
    exploit (a ring stood on its rim reads ~90deg deviation)."""
    cube: RigidObject = env.scene["cube"]
    q = cube.data.root_quat_w
    ramp = _lift_ramp(env, ramp_lo, ramp_hi)
    if not hasattr(env, "gr8_ref_quat"):
        env.gr8_ref_quat = torch.zeros(env.num_envs, 4, device=env.device)
        env.gr8_ref_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env.gr8_ref_valid &= ramp > 0.0
    cap = (~env.gr8_ref_valid) & (ramp >= capture_at)
    if cap.any():
        env.gr8_ref_quat[cap] = q[cap]
        env.gr8_ref_valid |= cap
    dot = (env.gr8_ref_quat * q).sum(dim=-1).abs().clamp(max=1.0)
    ang = 2.0 * torch.acos(dot)
    ang = torch.nan_to_num(ang, nan=0.0, posinf=max_angle).clamp(max=max_angle)
    return ang * ramp * env.gr8_ref_valid.float()


def object_orientation_hold(
    env: "ManagerBasedRLEnv",
    mode: str = "axis",
    sigma: float = 0.3,
    gate_mode: str = "any",
    ramp_lo: float = 0.01,
    ramp_hi: float = 0.05,
) -> torch.Tensor:
    """gr8b (operator 2026-08-31): keep-as-placed INCOME for all three objects —
    the positive twin of object_orientation_deviation (a penalty is the wrong
    shape per the Bible; the operator: "the policy should be rewarded for
    keeping that tube in its position"). Reference = the object's orientation
    at the EPISODE START, not at pickup: known at deploy (the object was
    placed; VisualModule measures it), no capture statefulness, and a tilted
    pickup is priced too. Income = exp(-(ang/sigma)^2) x touching x lift ramp,
    so it pays only while actually carrying (tiltgate ramps keep tipping
    worthless). mode "axis": angle between the object's z-axis now vs start —
    the tube/ring are rotationally symmetric, spin about their own axis is
    unobservable and must not be priced; "full": quaternion geodesic (cube)."""
    cube: RigidObject = env.scene["cube"]
    q = cube.data.root_quat_w
    if not hasattr(env, "gr8b_ref_quat"):
        env.gr8b_ref_quat = q.clone()
    fresh = env.episode_length_buf <= 1
    if fresh.any():
        env.gr8b_ref_quat[fresh] = q[fresh]
    if mode == "full":
        dot = (env.gr8b_ref_quat * q).sum(dim=-1).abs().clamp(max=1.0)
        ang = 2.0 * torch.acos(dot)
    else:
        ez = torch.zeros_like(q[:, :3])
        ez[:, 2] = 1.0
        z_now = math_utils.quat_apply(q, ez)
        z_ref = math_utils.quat_apply(env.gr8b_ref_quat, ez)
        ang = torch.acos((z_now * z_ref).sum(dim=-1).clamp(-1.0, 1.0))
    ang = torch.nan_to_num(ang, nan=torch.pi).clamp(max=torch.pi)
    if gate_mode == "closure":
        touching = _closure_touching(env)
    else:
        touching = (_pad_force_mags(env).sum(dim=-1) > 1.0).float()
    return torch.exp(-torch.square(ang / sigma)) * touching * _lift_ramp(env, ramp_lo, ramp_hi)


def hold_object_rim(
    env: "ManagerBasedRLEnv",
    rim_radius: float = 0.09,
    rim_z: float = 0.065,
    sigma: float = 0.06,
    gate_mode: str = "any",
    ramp_lo: float = 0.01,
    ramp_hi: float = 0.05,
) -> torch.Tensor:
    """gr8b rimhold A/B: hold_cube_bonus's dish with the distance measured to
    the NEAREST POINT OF THE (top) RIM CIRCLE instead of the object center.
    hold_cube's sigma-0.06 palm-to-CENTER kernel pays <=exp(-(R/sigma)^2)~=0.1
    on an R=9 cm rim object BY GEOMETRY (operator: "it barely fires... maybe
    the reward is built around the dimensions of the cube" — exactly). In the
    object frame: palm -> (rho, z); d = sqrt((rho - R)^2 + (z - rim_z)^2).
    Same 5%/95% touching/lift-ramp gate as hold_cube_bonus."""
    _buffers(env)
    cube: RigidObject = env.scene["cube"]
    p_pos, _ = _palm_pose(env)
    rel = math_utils.quat_apply_inverse(cube.data.root_quat_w, p_pos - cube.data.root_pos_w)
    rho = rel[:, :2].norm(dim=-1)
    d = torch.sqrt(torch.square(rho - rim_radius) + torch.square(rel[:, 2] - rim_z))
    d = torch.nan_to_num(d, nan=1e3, posinf=1e3)
    if gate_mode == "closure":
        touching = _closure_touching(env)
    else:
        touching = (_pad_force_mags(env).sum(dim=-1) > 1.0).float()
    gate = 0.05 * touching + 0.95 * touching * _lift_ramp(env, ramp_lo, ramp_hi)
    return torch.exp(-torch.square(d / sigma)) * gate


def approach_cube_bonus(env: "ManagerBasedRLEnv", sigma: float = 0.3) -> torch.Tensor:
    """gr6b_retry: the reach-back gradient. hold_cube's sigma 0.06 kernel is
    flat-zero at 30 cm, so after a slide the rational move was dangling at
    park (+2/s) — nothing paid for closing the distance. Wide kernel, small
    weight, and x(1 - touching) so it never double-pays while gripping."""
    cube: RigidObject = env.scene["cube"]
    p_pos, _ = _palm_pose(env)
    touching = (_pad_force_mags(env).sum(dim=-1) > 1.0).float()
    d = (cube.data.root_pos_w - p_pos).norm(dim=-1)
    return torch.exp(-((d / sigma) ** 2)) * (1.0 - touching)


def reset_cube_retry(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    prob: float = 0.0,
    dist_range: tuple[float, float] = (0.10, 0.20),
) -> None:
    """gr6b_retry seeding (the dp5 lean-seeding lesson): with `prob`, start
    the episode with the cube ON THE TABLE 10-20 cm away from the hand — the
    retry state the policy otherwise never visits (episodes begin with the
    hand above the cube per the production handoff, which the other 80% keep).
    Updates cube_start_pos_w for moved envs so the hold target sits above
    where the cube ACTUALLY is."""
    if prob <= 0.0 or len(env_ids) == 0:
        return
    cube: RigidObject = env.scene["cube"]
    platform = env.scene["platform"]
    picked = env_ids[torch.rand(len(env_ids), device=env.device) < prob]
    if len(picked) == 0:
        return
    d = dist_range[0] + torch.rand(len(picked), device=env.device) * (dist_range[1] - dist_range[0])
    ang = torch.rand(len(picked), device=env.device) * 2.0 * torch.pi
    root = cube.data.root_state_w[picked].clone()
    root[:, 0] += d * torch.cos(ang)
    root[:, 1] += d * torch.sin(ang)
    # clamp onto the platform footprint (0.45 x 0.45, 2 cm margin)
    pxy = platform.data.root_pos_w[picked, :2]
    root[:, 0] = root[:, 0].clamp(pxy[:, 0] - 0.205, pxy[:, 0] + 0.205)
    root[:, 1] = root[:, 1].clamp(pxy[:, 1] - 0.205, pxy[:, 1] + 0.205)
    root[:, 7:] = 0.0
    cube.write_root_pose_to_sim(root[:, :7], picked)
    cube.write_root_velocity_to_sim(root[:, 7:], picked)
    if hasattr(env, "cube_start_pos_w"):
        env.cube_start_pos_w[picked] = root[:, :3]


def mount_orbit(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    amp_range: tuple[float, float] = (0.0, 0.0),
    freq_range: tuple[float, float] = (0.1, 0.4),
) -> None:
    """gr6b_transport: continuous RANDOMIZED-PHASE mount motion — the arm
    base rides a slow per-env orbit (xy sinusoids, independent phases), so a
    held cube must survive being TRANSPORTED. Interval-timed wobble was the
    gr5b armwobble exploit (time-locked feedforward); random phase+freq per
    env makes the motion unpredictable from episode time. amp 0 = no-op
    (trunk default; the transport job turns it on via set_param)."""
    # harden vs renderer mishaps: a malformed set_param once delivered a STRING
    # here (unquoted JSON list word-split by bash, 2026-08-19) — fail loudly on
    # anything non-numeric instead of comparing str to float.
    amp_range = tuple(float(v) for v in amp_range)
    freq_range = tuple(float(v) for v in freq_range)
    if amp_range[1] <= 0.0:
        return
    robot = env.scene["robot"]
    if not hasattr(env, "_orbit_amp"):
        n = env.num_envs
        env._orbit_amp = torch.zeros(n, device=env.device)
        env._orbit_freq = torch.zeros(n, 2, device=env.device)
        env._orbit_phase = torch.zeros(n, 2, device=env.device)
    fresh = env._orbit_amp[env_ids] == 0.0
    if fresh.any():
        ids = env_ids[fresh]
        env._orbit_amp[ids] = amp_range[0] + torch.rand(len(ids), device=env.device) * (amp_range[1] - amp_range[0])
        env._orbit_freq[ids] = freq_range[0] + torch.rand(len(ids), 2, device=env.device) * (freq_range[1] - freq_range[0])
        env._orbit_phase[ids] = torch.rand(len(ids), 2, device=env.device) * 2.0 * torch.pi
    t = (env.episode_length_buf[env_ids] * env.step_dt).unsqueeze(-1)
    off = env._orbit_amp[env_ids].unsqueeze(-1) * torch.sin(
        2.0 * torch.pi * env._orbit_freq[env_ids] * t + env._orbit_phase[env_ids]
    )
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    root[:, :2] += off
    robot.write_root_pose_to_sim(root[:, :7], env_ids)
