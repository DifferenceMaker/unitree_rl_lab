"""Left-right mirror augmentation for the H1-2 walk line (lm4, paper #226).

Called by rsl_rl's PPO as ``symmetry_cfg.data_augmentation_func``:
``func(env=..., obs=TensorDict|None, actions=Tensor|None)`` and must return
``(obs_aug, actions_aug)`` with the ORIGINAL batch first, mirrored second
(rsl_rl assumes ``out[:batch]`` is the input, ppo.py symmetry block).

The permutation/sign maps are built LAZILY FROM THE LIVE ENV, not hardcoded:
the same observation vector contains TWO joint orderings (q/dq/effort use
BODY_JOINT_REGEX with preserve_order=True; last_action and the action space
use plain asset breadth-first order), so any hardcoded index table is one
refactor away from silent garbage. Only the per-joint-TYPE signs are pinned
here — they were machine-checked by an FK probe on the training asset
(2026-08-13, free-fall perturbation of every L/R pair, y-mirrored body-layout
match): every *_pitch/knee pair mirrors with +1; every *_roll/*_yaw pair and
the torso yaw with -1. Same-axis URDF conventions throughout, and the default
stance is exactly L/R-symmetric (all pair defaults equal), so pure perm*sign
is correct for joint_pos_rel, joint_vel, effort, and pre-scale actions alike.

Any obs term this module does not recognize -> hard RuntimeError. That is the
contract guard: reusing this func on a trunk with different obs (e.g. one
that still carries arm_pose_command) must fail loudly, not mirror nonsense.
"""

from __future__ import annotations

import torch

# FK-probed mirror signs by joint type (see module docstring).
_MINUS = ("_roll_", "_yaw_", "torso")
_PLUS = ("_pitch_", "_knee_")

# How to mirror each known obs term. "joints_obs"/"joints_act" use the two
# resolved joint orderings; 3-vectors get explicit sign patterns.
#   linear (x,y,z) -> (x,-y,z); angular rate (pseudovector) -> (-x,y,-z);
#   command (vx,vy,wz) -> (vx,-vy,-wz).
_TERM_KIND = {
    "base_lin_vel": ("signs", (1.0, -1.0, 1.0)),
    "base_ang_vel": ("signs", (-1.0, 1.0, -1.0)),
    "projected_gravity": ("signs", (1.0, -1.0, 1.0)),
    "velocity_commands": ("signs", (1.0, -1.0, -1.0)),
    "joint_pos_rel": ("joints_obs", None),
    "joint_vel_rel": ("joints_obs", None),
    "joint_effort": ("joints_obs", None),
    "last_action": ("joints_act", None),
}

_cache: dict = {}


def _joint_sign(name: str) -> float:
    if any(k in name for k in _MINUS):
        return -1.0
    if any(k in name for k in _PLUS):
        return 1.0
    raise RuntimeError(f"[symmetry] joint '{name}' has no probed mirror sign")


def _mirror_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name  # centerline (torso)


def _perm_sign(names: list[str]) -> tuple[list[int], list[float]]:
    """perm[i] = index of i's mirror partner; sign[i] = mirror sign of joint i."""
    perm = [names.index(_mirror_name(n)) for n in names]  # raises if missing
    sign = [_joint_sign(n) for n in names]
    # self-check: involution, and partner signs agree
    for i, j in enumerate(perm):
        assert perm[j] == i and sign[i] == sign[j], f"mirror map broken at {names[i]}"
    return perm, sign


def _build(env) -> dict:
    env = getattr(env, "unwrapped", env)
    robot = env.scene["robot"]

    act_term = env.action_manager.get_term("JointPositionAction")
    act_names = [robot.data.joint_names[i] for i in act_term._joint_ids]

    om = env.observation_manager
    idx = om.active_terms["policy"].index("joint_pos_rel")
    obs_asset_cfg = om._group_obs_term_cfgs["policy"][idx].params["asset_cfg"]
    obs_names = [robot.data.joint_names[i] for i in obs_asset_cfg.joint_ids]

    perm_o, sign_o = _perm_sign(obs_names)
    perm_a, sign_a = _perm_sign(act_names)
    n_act = len(act_names)

    maps: dict = {"n_act": n_act}
    # action-space map
    maps["action"] = (
        torch.tensor(perm_a, dtype=torch.long, device=env.device),
        torch.tensor(sign_a, device=env.device),
    )

    # one flat (perm, sign) per obs group, assembled term-by-term
    for group in env.observation_manager.active_terms.keys():
        names = env.observation_manager.active_terms[group]
        dims = [int(torch.prod(torch.tensor(d))) for d in env.observation_manager.group_obs_term_dim[group]]
        perm: list[int] = []
        sign: list[float] = []
        off = 0
        for name, dim in zip(names, dims):
            if name not in _TERM_KIND:
                raise RuntimeError(
                    f"[symmetry] obs term '{group}/{name}' has no mirror rule — "
                    "this trunk is not what the mirror map was written for"
                )
            kind, arg = _TERM_KIND[name]
            if kind == "signs":
                assert dim == 3, f"{name}: expected dim 3, got {dim}"
                perm += [off, off + 1, off + 2]
                sign += list(arg)
            else:
                p, s = (perm_o, sign_o) if kind == "joints_obs" else (perm_a, sign_a)
                assert dim == len(p), f"{name}: dim {dim} != joint map {len(p)}"
                perm += [off + k for k in p]
                sign += s
            off += dim
        maps[group] = (
            torch.tensor(perm, dtype=torch.long, device=env.device),
            torch.tensor(sign, device=env.device),
        )
    return maps


def _mirror(t: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    return t[..., perm] * sign


def mirror_h1_2_walk(env, obs=None, actions=None):
    """Data-augmentation func for RslRlSymmetryCfg (lm4 trunk, 27 joints)."""
    key = id(getattr(env, "unwrapped", env))
    if key not in _cache:
        _cache.clear()  # single live env per process
        _cache[key] = _build(env)
    maps = _cache[key]

    obs_out = None
    if obs is not None:
        mirrored = obs.clone()
        for group in obs.keys():
            perm, sign = maps[group]
            mirrored[group] = _mirror(obs[group], perm, sign)
        obs_out = torch.cat([obs, mirrored], dim=0)

    act_out = None
    if actions is not None:
        perm, sign = maps["action"]
        act_out = torch.cat([actions, _mirror(actions, perm, sign)], dim=0)

    return obs_out, act_out
