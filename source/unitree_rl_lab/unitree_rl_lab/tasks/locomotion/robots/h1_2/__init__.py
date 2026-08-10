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
