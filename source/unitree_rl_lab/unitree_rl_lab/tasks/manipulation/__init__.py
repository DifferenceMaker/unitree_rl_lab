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

# gr5c: the SAME Arm7 environment, with the entropy bonus off. A separate id,
# not a hydra override, because the queue hardcodes `-- --seed N` and has no
# extras hook — an override passed there would be SILENTLY IGNORED and the run
# would train at entropy_coef 0.01 while claiming otherwise. As a task id it is
# verifiable after the fact in the harvested params/agent.yaml, which is this
# project's ground truth. See GraspNoEntropyPPORunnerCfg for the autopsy.
gym.register(
    id="Unitree-H1_2-Grasp-Arm7-QS",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg_arm:RobotEnvCfgArm7",
        "play_env_cfg_entry_point": f"{__name__}.grasp_env_cfg_arm:RobotEnvCfgArm7",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GraspNoEntropyPPORunnerCfg",
    },
)

# gr5d: Arm7 with entropy 0.0 AND a fixed lr. QS's adaptive scheduler cut lr to
# the 1e-5 floor once sigma got small (KL ~ dmu^2/2sigma^2) and froze learning
# at ~iter 1500 — the runaway loop running backwards. See
# GraspNoEntropyFixedLRPPORunnerCfg for the measured trajectory.
# gr6: the permanent-table task (hold the cube at the hand's spawn point;
# retract removed). Two interfaces: Table = joint-space arm (isolates the task
# change), TS = task-space d(pose) through the DLS IK action (isolates the
# interface change; colleague-convergent).
gym.register(
    id="Unitree-H1_2-Grasp-Arm7Table-QF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg_arm:RobotEnvCfgArm7Table",
        "play_env_cfg_entry_point": f"{__name__}.grasp_env_cfg_arm:RobotEnvCfgArm7Table",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GraspNoEntropyFixedLRPPORunnerCfg",
    },
)
gym.register(
    id="Unitree-H1_2-Grasp-Arm7TS-QF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg_arm:RobotEnvCfgArm7TaskSpace",
        "play_env_cfg_entry_point": f"{__name__}.grasp_env_cfg_arm:RobotEnvCfgArm7TaskSpace",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GraspNoEntropyFixedLRPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-H1_2-Grasp-Arm7-QF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp_env_cfg_arm:RobotEnvCfgArm7",
        "play_env_cfg_entry_point": f"{__name__}.grasp_env_cfg_arm:RobotEnvCfgArm7",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:GraspNoEntropyFixedLRPPORunnerCfg",
    },
)
