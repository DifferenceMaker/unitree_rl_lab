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

@configclass
class QueuePPORunnerCfg(BasePPORunnerCfg):
    """Queue tasks share the base experiment_name so warmstart from
    milestones/<slug> resolves under logs/rsl_rl/unitree_h1_2_balance/."""
    experiment_name = "unitree_h1_2_balance"

@configclass
class WalkPPORunnerCfg(BasePPORunnerCfg):
    """Locomotion (lm line) runner.

    Pins experiment_name so BOTH Unitree-H1_2-Walk and Unitree-H1_2-Walk-Q log
    into logs/rsl_rl/unitree_h1_2_walk/. Without this, cli_args derives the name
    from the task id (task.lower().replace("-","_")), so the -Q variant would
    land in unitree_h1_2_walk_q/ and later lm runs could not warmstart from
    earlier ones by run name. Same reason V5PPORunnerCfg/QueuePPORunnerCfg pin
    "unitree_h1_2_balance".

    Deliberately NOT sharing the balance experiment_name: 27-action walk is a
    separate policy line and deploy path from the 13-action balance/desk head.
    """

    experiment_name = "unitree_h1_2_walk"


@configclass
class LieDownPPORunnerCfg(BasePPORunnerCfg):
    """Safety lie-down (sd line) runner.

    Own experiment_name / log dir: 27-action but a different action TRANSFORM
    (relative-to-measured-q, not default-offset) and reward economy — never
    warmstart from or into the balance/walk lineages.
    """

    experiment_name = "unitree_h1_2_liedown"


@configclass
class GraspPPORunnerCfg(BasePPORunnerCfg):
    """Inspire FTP tactile grasp (gr line). Own experiment_name so Grasp and
    Grasp-Q share one log dir (the WalkPPORunnerCfg pattern) and never mix with
    the balance lineage. Smaller net: 29-dim actor obs does not need 512."""

    experiment_name = "unitree_h1_2_grasp"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
