import gymnasium as gym

gym.register(
    id="Unitree-H1_2-Balance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H1_2-Balance-V5A",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_v5:RobotEnvCfgV5A",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_v5:RobotPlayEnvCfgV5A",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:V5PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H1_2-Balance-V5B",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_v5:RobotEnvCfgV5B",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_v5:RobotPlayEnvCfgV5B",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:V5PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H1_2-Balance-V5C",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_v5:RobotEnvCfgV5C",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_v5:RobotPlayEnvCfgV5C",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:V5PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H1_2-Balance-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_queue:RobotEnvCfgQueue",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_queue:RobotPlayEnvCfgQueue",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueuePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H1_2-Balance-IK",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_ik:RobotEnvCfgIK",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_ik:RobotPlayEnvCfgIK",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# lm3: commanded-velocity walking on the p13c economy, 27 actions, from
# scratch (see lm3_env_cfg.py). WalkPPORunnerCfg -> logs under unitree_h1_2_walk.
gym.register(
    id="Unitree-H1_2-LM3-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm3_env_cfg:RobotEnvCfgLM3",
        "play_env_cfg_entry_point": f"{__name__}.lm3_env_cfg:RobotPlayEnvCfgLM3",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:WalkPPORunnerCfg",
    },
)

# lm4: lm3 minus the arm command channel (obs 104 -> 90, SCRATCH only),
# + mirror loss (paper #226) + command-magnitude curriculum. The -LCP variant
# swaps in LCPPPO (paper #14 gradient penalty) and deletes action_rate.
gym.register(
    id="Unitree-H1_2-LM4-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm4_env_cfg:RobotEnvCfgLM4",
        "play_env_cfg_entry_point": f"{__name__}.lm4_env_cfg:RobotPlayEnvCfgLM4",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4PPORunnerCfg",
    },
)

# lm4b: lm4 trunk + curriculum gate_frac 0.5 (the survival-coupled 0.8 gate
# never fires below ~80% survival x quality). Two single-delta runs: track
# weights via job set_weight (LM4B-Q), mirror 0.1 via runner cfg (LM4B-M01-Q).
gym.register(
    id="Unitree-H1_2-LM4B-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm4b_env_cfg:RobotEnvCfgLM4B",
        "play_env_cfg_entry_point": f"{__name__}.lm4b_env_cfg:RobotPlayEnvCfgLM4B",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H1_2-LM4B-M01-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm4b_env_cfg:RobotEnvCfgLM4B",
        "play_env_cfg_entry_point": f"{__name__}.lm4b_env_cfg:RobotPlayEnvCfgLM4B",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4BMirror01PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H1_2-LM4B-LCP-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm4b_env_cfg:RobotEnvCfgLM4B",
        "play_env_cfg_entry_point": f"{__name__}.lm4b_env_cfg:RobotPlayEnvCfgLM4B",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4BLcpMirror01PPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H1_2-LM4B-LCPR-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm4b_env_cfg:RobotEnvCfgLM4B",
        "play_env_cfg_entry_point": f"{__name__}.lm4b_env_cfg:RobotPlayEnvCfgLM4B",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4BLcpRetrofitPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H1_2-LM4-LCP-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm4_env_cfg:RobotEnvCfgLM4LCP",
        "play_env_cfg_entry_point": f"{__name__}.lm4_env_cfg:RobotPlayEnvCfgLM4",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4LcpPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H1_2-Balance-QIK",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_queue_ik:RobotEnvCfgQueueIK",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_queue_ik:RobotPlayEnvCfgQueueIK",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueuePPORunnerCfg",
    },
)

# Balance-QIK under the LCP retrofit runner (p13e_lcpr, 2026-09-07): identical
# ENV to Balance-QIK — only the algorithm changes (LCPPPO, which itself chains
# GuardedPPO). DeskLcpRetrofitPPORunnerCfg already extends QueuePPORunnerCfg,
# the runner this task uses, so it is reusable verbatim: no symmetry_cfg (the
# mirror map is the walk obs contract), action_rate kept as the sigma anchor.
gym.register(
    id="Unitree-H1_2-Balance-QIK-LCPR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_queue_ik:RobotEnvCfgQueueIK",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_queue_ik:RobotPlayEnvCfgQueueIK",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:DeskLcpRetrofitPPORunnerCfg",
    },
)

gym.register(
    # p13f (2026-09-09): QIK env, FIXED lr 3e-4 runner. The adaptive-KL
    # scheduler A/B for the clip-off action-price runs (see
    # QueueFixedLRPPORunnerCfg). Same env cfg, same experiment_name, so
    # warmstart/harvest paths are identical to Balance-QIK.
    id="Unitree-H1_2-Balance-QIK-FIXLR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_queue_ik:RobotEnvCfgQueueIK",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_queue_ik:RobotPlayEnvCfgQueueIK",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueueFixedLRPPORunnerCfg",
    },
)

gym.register(
    # p14b (2026-09-16): p14_kitchen promoted to parent + TRUNK. Balance-QIK-FIXLR
    # with the p14_kitchen.job apply_edits baked (balance_env_cfg_kitchen), job
    # overrides applied last -> every p14b job is a pure delta on what kitchen
    # trained on. Same runner cfg / experiment_name as QIK-FIXLR, so p14_kitchen
    # warmstarts and harvest paths are unchanged.
    id="Unitree-H1_2-Balance-Kitchen-FIXLR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_kitchen:RobotEnvCfgKitchen",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_kitchen:RobotPlayEnvCfgKitchen",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueueFixedLRPPORunnerCfg",
    },
)

gym.register(
    # STANCE trunk (2026-09-17) — Kitchen with feet_too_near.threshold = 0.25 baked, i.e. the
    # p14b_stance delta promoted to parent after it solved the inward creep on hardware. p14c
    # jobs are therefore pure deltas on what stance trained on; a job that merely warmstarted
    # from the stance milestone WITHOUT this trunk would silently revert to kitchen's 0.22,
    # because terms do not travel with a checkpoint. Same runner cfg / experiment_name as
    # Kitchen-FIXLR, so p14b_stance warmstarts and harvest paths are unchanged.
    id="Unitree-H1_2-Balance-Stance-FIXLR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_stance:RobotEnvCfgStance",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_stance:RobotPlayEnvCfgStance",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueueFixedLRPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H1_2-Balance-Desk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk:RobotEnvCfgDesk",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk:RobotPlayEnvCfgDesk",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueuePPORunnerCfg",
    },
)

# --- Locomotion (lm line, 2026-07-28) --------------------------------
# 27-action walk (13 leg+torso + 14 arm). SEPARATE deploy path from the
# 13-action balance/desk line by design — arms swing for human-like gait.
# experiment_name derives to unitree_h1_2_walk (own log dir, no mixing
# with the balance warmstart lineage).
gym.register(
    id="Unitree-H1_2-Walk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.walk_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:WalkPPORunnerCfg",
    },
)

# Queue-trainable variant: reads QUEUE_JOB_JSON + pins the SYM asset.
gym.register(
    id="Unitree-H1_2-Walk-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:RobotEnvCfgWalkQueue",
        "play_env_cfg_entry_point": f"{__name__}.walk_env_cfg:RobotPlayEnvCfgWalkQueue",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:WalkPPORunnerCfg",
    },
)

# --- Safety lie-down (sd line, 2026-07-28) ---------------------------
# 27-action RELATIVE joint-position policy (HoST-style β bound): on trigger
# (overheat/low battery) the robot lowers itself slowly to supine. Separate
# deploy transform from every other line — target = measured q + Δ.
gym.register(
    id="Unitree-H1_2-LieDown",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liedown_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.liedown_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LieDownPPORunnerCfg",
    },
)

# dp5: the desk trunk AS A TASK (every settled dp4 conclusion baked in, so a
# dp5 job carries only its own axis). A new id, not a change to Balance-Desk —
# the historical desk policies replay their previews against that task.
gym.register(
    id="Unitree-H1_2-Balance-Desk5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk5:RobotEnvCfgDesk5",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk5:RobotPlayEnvCfgDesk5",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueuePPORunnerCfg",
    },
)

# dp5b: dp5 with the moving desk/anchor removed and the arm targets lifted out
# of the slab. Again a new id, not a mutation of Desk5 — the seven dp5 previews
# replay against Desk5 and must keep showing the world those policies trained
# in (teleporting slab, targets inside the table).
gym.register(
    id="Unitree-H1_2-Balance-Desk5b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk5b:RobotEnvCfgDesk5b",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk5b:RobotPlayEnvCfgDesk5b",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueuePPORunnerCfg",
    },
)

# lm5: the lm4e_wall_hips recipe trunked (operator 2026-08-20, trunk-is-task) —
# the unparked parent. Mirror 0.1 runner (wall_hips ran LM4B-M01-Q).
gym.register(
    id="Unitree-H1_2-LM5-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm5_env_cfg:RobotEnvCfgLM5",
        "play_env_cfg_entry_point": f"{__name__}.lm5_env_cfg:RobotPlayEnvCfgLM5",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4BMirror01PPORunnerCfg",
    },
)

# lm5d_combo TRUNK (2026-08-28): LM5 + _make_lm5d_combo baked; lm5e jobs warmstart
# milestones/lm5d_combo_2026-08-27 on this task and carry only their deltas.
gym.register(
    id="Unitree-H1_2-LM5-C",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm5_env_cfg:RobotEnvCfgLM5C",
        "play_env_cfg_entry_point": f"{__name__}.lm5_env_cfg:RobotPlayEnvCfgLM5C",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4BMirror01PPORunnerCfg",
    },
)

# lm5 LCP retrofit: LM5 env + the reduced-dose retrofit runner (lcp_num_steps=1,
# action_rate KEPT = sigma anchor) — the operator's clean A/B: wall_hips vs
# wall_hips+retrofit, same genes, retrofit the only difference.
gym.register(
    id="Unitree-H1_2-LM5-LCPR-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm5_env_cfg:RobotEnvCfgLM5",
        "play_env_cfg_entry_point": f"{__name__}.lm5_env_cfg:RobotPlayEnvCfgLM5",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4BLcpRetrofitPPORunnerCfg",
    },
)

# dp5e_lcpr: Desk5b env + the desk LCP-retrofit runner (no walk mirror).
gym.register(
    id="Unitree-H1_2-Balance-Desk5b-LCPR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk5b:RobotEnvCfgDesk5b",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk5b:RobotPlayEnvCfgDesk5b",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:DeskLcpRetrofitPPORunnerCfg",
    },
)

# Desk6: the lean-command trunk (+1 obs = contract break, SCRATCH-ONLY).
gym.register(
    id="Unitree-H1_2-Balance-Desk6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk6:RobotEnvCfgDesk6",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk6:RobotPlayEnvCfgDesk6",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueuePPORunnerCfg",
    },
)

# Desk6b: Desk6 + desk_penetration termination (dp6b; same 91-obs contract).
gym.register(
    id="Unitree-H1_2-Balance-Desk6b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk6:RobotEnvCfgDesk6b",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk6:RobotPlayEnvCfgDesk6b",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueuePPORunnerCfg",
    },
)

# Desk7: Desk6b + the dp7_hinge stack baked (hinge promoted to trunk,
# operator 2026-09-03; same 91-obs contract — hinge warmstarts stay valid).
gym.register(
    id="Unitree-H1_2-Balance-Desk7",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk7:RobotEnvCfgDesk7",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk7:RobotPlayEnvCfgDesk7",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueuePPORunnerCfg",
    },
)

gym.register(
    # dp8b (2026-09-11): Desk7 env, FIXED lr 3e-4 runner — the general line's
    # single biggest result, never transferred to the desk line. Every dp8 run
    # trained on `schedule: adaptive` at lr 1e-3; on the balance line the same
    # recipe under a fixed 3e-4 went 1400 -> 1781 with less than half the
    # limit-riding and the lowest action_rate of the wave (p13f_softfree30 vs
    # _fixlr, identical deltas). NOTE the desk line has not shown the p13e/p13f
    # detonation, so this is a bet on final QUALITY, not a fix for a collapse.
    # Same env cfg and experiment_name as Balance-Desk7, so warmstart/harvest
    # paths are identical and dp8 milestones warmstart it unchanged.
    id="Unitree-H1_2-Balance-Desk7-FIXLR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk7:RobotEnvCfgDesk7",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk7:RobotPlayEnvCfgDesk7",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueueFixedLRPPORunnerCfg",
    },
)

# Desk8-Work (2026-09-30): the WORK policy trunk -- Desk7 + the dp8c_yaw heading stack (93 obs)
# + the BALANCE carry list, transit teleports OFF, anchor frozen, BALANCE push regime and
# HipsArm arm envelope, lean-aware base_height back at -15, hand790 body. SCRATCH.
gym.register(
    id="Unitree-H1_2-Balance-Desk8-Work",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk8:RobotEnvCfgDesk8Work",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk8:RobotPlayEnvCfgDesk8Work",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueueFixedLRPPORunnerCfg",
    },
)

# lm5-fsb: FastSAC Run B — LM5 world, holosoma's 10-term minimalist rewards
# (g1_29dof_loco_fast_sac preset verbatim). Run A isolates the algorithm;
# this isolates the reward philosophy. Trained via train_fastsac.py (the
# rsl_rl entry is only there so a PPO arm on the same rewards stays possible).
gym.register(
    id="Unitree-H1_2-LM5-FSB",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm5_fsb_env_cfg:RobotEnvCfgLM5FSB",
        "play_env_cfg_entry_point": f"{__name__}.lm5_fsb_env_cfg:RobotPlayEnvCfgLM5FSB",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueuePPORunnerCfg",
    },
)

# lm5-fsb2: FSB + holosoma's penalty curriculum (0.5x -> 1.0x ramp on episode
# length). Isolates the ONE recipe piece pilot B lacked.
gym.register(
    id="Unitree-H1_2-LM5-FSB2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm5_fsb_env_cfg:RobotEnvCfgLM5FSB2",
        "play_env_cfg_entry_point": f"{__name__}.lm5_fsb_env_cfg:RobotPlayEnvCfgLM5FSB2",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueuePPORunnerCfg",
    },
)

# LM6 (2026-09-28): the p14 balance conclusions unioned onto the LM5-C + lm5f_symonly recipe,
# real-hand body + Unitree armature table, trained FROM SCRATCH (operator: "a fresh start").
# Same runner as LM5-C (mirror 0.1, adaptive lr, GuardedPPO) -- fixed lr is a polish tool.
gym.register(
    id="Unitree-H1_2-LM6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm6_env_cfg:RobotEnvCfgLM6",
        "play_env_cfg_entry_point": f"{__name__}.lm6_env_cfg:RobotPlayEnvCfgLM6",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4BMirror01PPORunnerCfg",
    },
)

# HipsArm-FIXLR (2026-09-28): the p14g_hips_armfull recipe promoted to trunk -- Stance + tilt (std .05
# w3) + hips -1.0 + the full arm envelope baked, real-hand body. p14h jobs warmstart
# milestones/p14g_hips_armfull_2026-09-23 on this task and carry only their deltas. Same runner cfg /
# experiment_name as Stance-FIXLR, so warmstarts and harvest paths are unchanged.
gym.register(
    id="Unitree-H1_2-Balance-HipsArm-FIXLR",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_hipsarm:RobotEnvCfgHipsArm",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_hipsarm:RobotPlayEnvCfgHipsArm",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueueFixedLRPPORunnerCfg",
    },
)

# LM6B (2026-09-29): LM6 with the push / sustained-push curricula gated on SURVIVAL (>= 50 % time-outs
# over >= 4096 resets, >= 1000 iterations per level, first level held 1000 iterations) instead of the
# step clock that put 1.5 m/s pushes on a scratch walker by iteration ~1000 (lm6: every episode ended
# at the first push). Same runner as LM6.
gym.register(
    id="Unitree-H1_2-LM6B",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm6_env_cfg:RobotEnvCfgLM6B",
        "play_env_cfg_entry_point": f"{__name__}.lm6_env_cfg:RobotPlayEnvCfgLM6B",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4BMirror01PPORunnerCfg",
    },
)

# LM6C (2026-10-01): lm6b_sharp promoted to trunk (sharp economy baked) + gates the std .15 kernel can pass
# (push/sustained: tracking .20 + survival .60; velocity levels .25) + feet_slide -2.0. Parent lm6b_sharp_2026-09-29.
gym.register(
    id="Unitree-H1_2-LM6C",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm6_env_cfg:RobotEnvCfgLM6C",
        "play_env_cfg_entry_point": f"{__name__}.lm6_env_cfg:RobotPlayEnvCfgLM6C",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4BMirror01PPORunnerCfg",
    },
)

# Desk8b-Work (2026-10-05): Desk8-Work + the dp9 fixes -- shoulders/elbows added to undesired_contacts (-100),
# desk_hit re-added at -1.0 (hands on the desk discouraged, clamped), DESK push regime (20-30 s to 1.0 m/s,
# 30-50 s to 30 N). Parent by TRANSPLANT: p14g_hips_armfull_deskpad_2026-10-05 (87 -> 93 obs).
gym.register(
    id="Unitree-H1_2-Balance-Desk8b-Work",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk8b:RobotEnvCfgDesk8bWork",
        "play_env_cfg_entry_point": f"{__name__}.balance_env_cfg_desk8b:RobotPlayEnvCfgDesk8bWork",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:QueueFixedLRPPORunnerCfg",
    },
)

# LM7 (2026-10-06): lm5f_symonly (the walker) continued -- symonly's ledger + hand790/motor table + gait on
# the command + the linear box resumed at the parent's final; the lm6 and balance conclusions arrive as
# edit_raw genes (lm7_env_cfg.gene_*). Parent milestones/lm5f_symonly_resume_2026-08-31 (model_63421).
gym.register(
    id="Unitree-H1_2-LM7",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lm7_env_cfg:RobotEnvCfgLM7",
        "play_env_cfg_entry_point": f"{__name__}.lm7_env_cfg:RobotPlayEnvCfgLM7",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:LM4BMirror01PPORunnerCfg",
    },
)
