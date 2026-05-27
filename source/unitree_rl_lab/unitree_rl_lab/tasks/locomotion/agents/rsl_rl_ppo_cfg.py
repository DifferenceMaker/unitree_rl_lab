# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 25
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class V5PPORunnerCfg(BasePPORunnerCfg):
    """Override for V5A/B/C variants to share experiment_name with base task.
    
    Without this override, the V5A/B/C tasks default to experiment_name=""
    which auto-derives from task ID, creating separate log dirs
    (unitree_h1_2_balance_v5a/b/c). But train.sh's warmstart copies the
    milestone to the BASE experiment dir (unitree_h1_2_balance/), so the V5
    tasks can't find the warmstart checkpoint and fail at load time.
    
    Setting experiment_name explicitly here makes all three V5 variants
    share the base task's log dir, so warmstart works.
    """
    experiment_name = "unitree_h1_2_balance"
