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

# Real Inspire SDK motor order: angle_set[0..5]. Drivers in the URDF.
DRIVER_JOINTS = [
    "left_little_1_joint",   # 0
    "left_ring_1_joint",     # 1
    "left_middle_1_joint",   # 2
    "left_index_1_joint",    # 3
    "left_thumb_2_joint",    # 4  thumb bend
    "left_thumb_1_joint",    # 5  thumb rotation/yaw
]
# follower -> (driver, multiplier). Thumb chain resolved to the bend driver.
FOLLOWER_COUPLING = {
    "left_little_2_joint": ("left_little_1_joint", 1.0843),
    "left_ring_2_joint": ("left_ring_1_joint", 1.0843),
    "left_middle_2_joint": ("left_middle_1_joint", 1.0843),
    "left_index_2_joint": ("left_index_1_joint", 1.0843),
    "left_thumb_3_joint": ("left_thumb_2_joint", 0.8024),
    "left_thumb_4_joint": ("left_thumb_2_joint", 0.8024 * 0.9487),
}

PALM_BODY = "left_palm_force_sensor"
THUMB_PAD_PREFIX = "left_thumb_force_sensor"


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
        self._f2d = torch.tensor(
            [DRIVER_JOINTS.index(FOLLOWER_COUPLING[n][0]) for n in f_found], device=self.device
        )
        self._fmul = torch.tensor(
            [FOLLOWER_COUPLING[n][1] for n in f_found], device=self.device
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
        self._targets_f = (self._targets_d[:, self._f2d] * self._fmul).clamp(self._f_lo, self._f_hi)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._targets_d, joint_ids=self._driver_ids)
        self._asset.set_joint_position_target(self._targets_f, joint_ids=self._follower_ids)

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
        thumb = torch.tensor([n.startswith(THUMB_PAD_PREFIX) for n in names], device=env.device)
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
def reset_grasp_scene(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    palm_xy: tuple = (0.0, 0.12),
    xy_jitter: float = 0.03,
    gap_range: tuple = (0.02, 0.08),
    cube_height: float = 0.055,
    platform_thickness: float = 0.02,
    retract_time_range: tuple = (3.0, 5.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset: place the kinematic platform so the palm-to-cube-top gap is
    U[gap_range]; drop the cube on it with random yaw and xy jitter under the
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
    plat_pose[:, 0] = origins[:, 0] + palm_xy[0]
    plat_pose[:, 1] = origins[:, 1] + palm_xy[1]
    plat_pose[:, 2] = plat_top - platform_thickness / 2
    plat_pose[:, 3] = 1.0
    platform.write_root_pose_to_sim(plat_pose, env_ids=env_ids)

    cube_pose = torch.zeros(n, 7, device=dev)
    cube_pose[:, 0] = plat_pose[:, 0] + (torch.rand(n, device=dev) * 2 - 1) * xy_jitter
    cube_pose[:, 1] = plat_pose[:, 1] + (torch.rand(n, device=dev) * 2 - 1) * xy_jitter
    cube_pose[:, 2] = plat_top + cube_height / 2 + 0.002
    yaw = (torch.rand(n, device=dev) * 2 - 1) * torch.pi
    cube_pose[:, 3] = torch.cos(yaw / 2)
    cube_pose[:, 6] = torch.sin(yaw / 2)
    cube.write_root_pose_to_sim(cube_pose, env_ids=env_ids)
    cube.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    env.grasp_retract_t[env_ids] = retract_time_range[0] + torch.rand(n, device=dev) * (
        retract_time_range[1] - retract_time_range[0]
    )
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
def hold_cube_bonus(env: "ManagerBasedRLEnv", sigma: float = 0.06) -> torch.Tensor:
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
    touching = (_pad_force_mags(env).sum(dim=-1) > 1.0).float()
    gate = 0.05 * touching + 0.95 * env.grasp_retracted.float()
    return torch.exp(-((d / sigma) ** 2)) * gate


def pad_arrangement_bonus(env: "ManagerBasedRLEnv", force_thr: float = 0.5) -> torch.Tensor:
    """Dense pre-grasp shaping: 0.3 for any pad contact, +0.7 for FORCE
    CLOSURE (a thumb pad AND an opposing finger pad both loaded) — rewards
    grasps, not scoops."""
    f = _pad_force_mags(env)
    thumb_m, finger_m = _pad_masks(env)
    any_c = (f > force_thr).any(dim=-1).float()
    closure = ((f[:, thumb_m] > force_thr).any(dim=-1) & (f[:, finger_m] > force_thr).any(dim=-1)).float()
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
