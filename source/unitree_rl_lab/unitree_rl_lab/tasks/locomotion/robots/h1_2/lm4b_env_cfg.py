"""LM4B — the lm4 trunk with a REACHABLE command-curriculum gate. Operator
design 2026-08-14, the two-run A/B on why lm4 stalled.

lm4 post-mortem (video + ledger, 2026-08-14): stands OK, gait attempts visible
("one foot in front of the other"), arms zombie-flail; 96% of episodes end by
base_height at ~24 s (eplen plateaued 1186/3000 from mid-training); sigma
FROZEN at 0.96-0.99 for all 16k iters (cleargate annealed 0.99->0.77); mirror
loss (coeff 1.0, ~0.008) outweighs the surrogate (~0.003) 3-4x in the total
objective; tracking quality WHILE ALIVE ~64% of ceiling (0.76 actual vs 1.185
perfect-at-24s-survival). lin/ang_vel_levels never moved off their 0.4/0.3
starts.

The ONE shared trunk change vs lm4: curriculum gate_frac 0.8 -> 0.5. The stock
(unitree h1/g1/go2) gate normalizes the episode tracking sum by MAX episode
length, so it couples quality with survival — perfect tracking at 40% survival
reads 0.4*weight, and even a healthy walker at cleargate-quality (~52-64%)
never clears 0.8. At 0.5 the gate fires when tracking is genuinely decent AND
survival is high — the curriculum axis stops being dead weight.

The two runs on this trunk (single-delta each, deltas live OUTSIDE this file):
  * lm4b_track    — job set_weight: track_lin_vel_xy 3->10, track_ang_vel_z
                    1.5->5 (operator hypothesis: walking pressure vs alive-30
                    is the binding constraint).
  * lm4b_mirror01 — task Unitree-H1_2-LM4B-M01-Q (runner cfg
                    LM4BMirror01PPORunnerCfg, mirror_loss_coeff 1.0->0.1; an
                    agent-side param must be a task id, the gr5c hydra lesson)
                    (my hypothesis: mirror-loss dominance drowned the task
                    gradient — frozen sigma is the tell).

Read order when they land: survival (base_height terminations, eplen -> 3000),
sigma anneal (0.99 -> 0.7s), tracking quality, THEN whether lin_vel_levels
climbs past 0.4.
"""

from isaaclab.utils import configclass

from .balance_env_cfg import RobotEnvCfg, RobotPlayEnvCfg
from .balance_env_cfg_queue import _apply_overrides, _load_overrides
from .lm3_env_cfg import _make_lm3
from .lm4_env_cfg import _make_lm4


def _make_lm4b(cfg):
    cfg.curriculum.lin_vel_levels.params["gate_frac"] = 0.5
    cfg.curriculum.ang_vel_levels.params["gate_frac"] = 0.5


@configclass
class RobotEnvCfgLM4B(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_lm3(self)
        _make_lm4(self)
        _make_lm4b(self)
        _apply_overrides(self, _load_overrides())   # jobs win, applied last


@configclass
class RobotPlayEnvCfgLM4B(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_lm3(self)
        _make_lm4(self)
        _make_lm4b(self)
        self.commands.base_velocity.debug_vis = True
        _apply_overrides(self, _load_overrides())
