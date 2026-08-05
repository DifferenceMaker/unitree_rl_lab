"""Queue task — a single registered env whose parameters are overridden at
launch time from a per-job JSON file.

The queue scheduler (scripts/queue/queue_runner.py) writes a JSON of overrides
per job and points the env var QUEUE_JOB_JSON at it before launching training.
This config subclasses the base RobotEnvCfg (which carries the current p7 reward
set) and applies those overrides in __post_init__.

Override JSON schema (all keys optional):
{
  "job_name": "exp1_torso12_airtime",
  "set_weight":   {"torso_stability": -12.0, "feet_air_time_step": -2.0},
  "set_param":    {"upright_bonus.std": 0.025,
                   "add_base_mass.mass_distribution_params": [0.0, 0.0]},
  "add_reward":   {"upright_bonus": {"func": "upright_bonus", "weight": 1.0,
                                     "params": {"std": 0.025}}},
  "edit_raw":     ["self.rewards.flat_orientation_l2.weight = -3.0"]
}

- set_weight: shortcut for self.rewards.<name>.weight = value
- set_param:  dotted path "<term>.<param>" -> sets self.rewards.<term>.params[...]
              OR self.events.<term>.params[...] (auto-detected). For nested
              params use "<term>.<param>" where param may itself index a dict
              via "<term>.params.<key>" — see _apply_set_param.
- add_reward: attach a NEW RewTerm to self.rewards (e.g. add upright_bonus that
              isn't in the base set). func is resolved from mdp.
- edit_raw:   arbitrary python executed with `self` in scope, last-resort.

Everything is best-effort with loud logging so a bad override fails the run
visibly rather than silently training the wrong thing.
"""
import json
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg


def _resolve_func(name):
    """Resolve a reward/event function name: locomotion mdp first, then the
    manipulation grasp_mdp (gr4 add_reward funcs live there — the gr4 batch
    failed to boot on this lookup, 2026-08-05)."""
    fn = getattr(mdp, name, None)
    if fn is None:
        try:
            from unitree_rl_lab.tasks.manipulation import grasp_mdp as _gm
            fn = getattr(_gm, name, None)
        except Exception:
            fn = None
    if fn is None:
        raise ValueError(f"[QUEUE] add_reward func '{name}' not found in mdp/grasp_mdp")
    return fn


def _apply_overrides(cfg_self, overrides: dict):
    job = overrides.get("job_name", "<unnamed>")
    print(f"[QUEUE] applying overrides for job '{job}'")

    # --- set_weight: self.rewards.<name>.weight = value ---
    for name, val in overrides.get("set_weight", {}).items():
        term = getattr(cfg_self.rewards, name, None)
        if term is None:
            raise ValueError(f"[QUEUE] set_weight: reward term '{name}' not in base cfg")
        term.weight = float(val)
        print(f"[QUEUE]   set_weight  rewards.{name}.weight = {val}")

    # --- set_param: "<term>.<param>" on rewards OR events ---
    for path, val in overrides.get("set_param", {}).items():
        term_name, _, param_key = path.partition(".")
        if not param_key:
            raise ValueError(f"[QUEUE] set_param path '{path}' must be '<term>.<param>'")
        # find the term on rewards first, then events, then curriculum
        container = None
        for cname in ("rewards", "events", "curriculum"):
            c = getattr(cfg_self, cname, None)
            if c is not None and getattr(c, term_name, None) is not None:
                container = c
                break
        if container is None:
            raise ValueError(f"[QUEUE] set_param: term '{term_name}' not found on rewards/events/curriculum")
        term = getattr(container, term_name)
        # std lives as a direct kwarg-style param in params dict for reward fns;
        # for events it's also params[...]. Both store under term.params.
        if not hasattr(term, "params") or term.params is None:
            raise ValueError(f"[QUEUE] set_param: term '{term_name}' has no params dict")
        # lists in json -> tuples (Isaac expects tuples for ranges)
        if isinstance(val, list):
            val = tuple(val)
        term.params[param_key] = val
        print(f"[QUEUE]   set_param   {container.__class__.__name__}.{term_name}.params[{param_key}] = {val}")

    # --- add_reward: attach a NEW RewTerm ---
    for name, spec in overrides.get("add_reward", {}).items():
        fn = _resolve_func(spec["func"])
        params = spec.get("params", {})
        # convert any nested SceneEntityCfg specs passed as {"_scene": {...}}
        params = _hydrate_params(params)
        term = RewTerm(func=fn, weight=float(spec.get("weight", 1.0)), params=params)
        setattr(cfg_self.rewards, name, term)
        print(f"[QUEUE]   add_reward  rewards.{name} = RewTerm({spec['func']}, weight={spec.get('weight',1.0)}, params={params})")

    # --- edit_raw: arbitrary python with `self` bound to the cfg ---
    for line in overrides.get("edit_raw", []):
        print(f"[QUEUE]   edit_raw    {line}")
        exec(line, {"mdp": mdp, "RewTerm": RewTerm, "EventTerm": EventTerm,
                    "ObsTerm": ObsTerm, "SceneEntityCfg": SceneEntityCfg},
             {"self": cfg_self})

    print(f"[QUEUE] overrides applied for job '{job}'")


def _hydrate_params(params: dict):
    """Convert JSON-friendly param specs into Isaac objects.

    A param value of the form {"_scene": {"name": "robot", "body_names": [...]}}
    becomes SceneEntityCfg(name="robot", body_names=[...]). Lists stay lists
    unless the consumer wants tuples (handled per-fn). Everything else passes
    through.
    """
    out = {}
    for k, v in params.items():
        if isinstance(v, dict) and "_scene" in v:
            out[k] = SceneEntityCfg(**v["_scene"])
        else:
            out[k] = v
    return out


def _load_overrides():
    path = os.environ.get("QUEUE_JOB_JSON")
    if not path:
        print("[QUEUE] WARNING: QUEUE_JOB_JSON not set — running base cfg unchanged.")
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[QUEUE] QUEUE_JOB_JSON points at missing file: {path}")
    with open(path) as f:
        ov = json.load(f)
    print(f"[QUEUE] loaded overrides from {path}")
    return ov


@configclass
class RobotEnvCfgQueue(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_overrides(self, _load_overrides())


@configclass
class RobotPlayEnvCfgQueue(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_overrides(self, _load_overrides())
