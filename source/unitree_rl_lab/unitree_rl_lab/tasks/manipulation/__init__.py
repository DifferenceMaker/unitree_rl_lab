"""Manipulation tasks (gr line): Inspire FTP tactile grasping."""

import gymnasium as gym

gym.register(
    id="Unitree-H1_2-Grasp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.grasp_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
    },
)

# Queue-trainable variant: reads QUEUE_JOB_JSON (same helpers as Balance-Desk).
gym.register(
    id="Unitree-H1_2-Grasp-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg:RobotEnvCfgGraspQueue",
        "play_env_cfg_entry_point": f"{__name__}.grasp_env_cfg:RobotPlayEnvCfgGraspQueue",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
    },
)

# gr3b: residual-wrist variant — floating hand base + 6-DoF WristPoseAction
# (12 actions, 35-dim actor obs). Same PPO cfg; dims are introspected.
gym.register(
    id="Unitree-H1_2-Grasp-Wrist-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg:RobotEnvCfgGraspWristQueue",
        "play_env_cfg_entry_point": f"{__name__}.grasp_env_cfg:RobotEnvCfgGraspWristQueue",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
    },
)

# gr5b: arm-in-the-loop variants (2026-08-06) — hand on a real H1-2 arm chain,
# policy owns arm joints + fingers; queue overrides applied last.
gym.register(
    id="Unitree-H1_2-Grasp-Wrist3-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg_arm:RobotEnvCfgWrist3",
        "play_env_cfg_entry_point": f"{__name__}.grasp_env_cfg_arm:RobotEnvCfgWrist3",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
    },
)
gym.register(
    id="Unitree-H1_2-Grasp-Arm7-Q",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg_arm:RobotEnvCfgArm7",
        "play_env_cfg_entry_point": f"{__name__}.grasp_env_cfg_arm:RobotEnvCfgArm7",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GraspPPORunnerCfg",
    },
)
