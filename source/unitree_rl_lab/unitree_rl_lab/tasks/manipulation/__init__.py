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
