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


@configclass
class GraspNoEntropyPPORunnerCfg(GraspPPORunnerCfg):
    """gr5c: GraspPPORunnerCfg with the entropy bonus OFF (entropy_coef 0.0).

    Every gr5b run had a RUNAWAY policy sigma (measured 2026-08-11): mean_std
    1.0 -> 479 (gr5b_arm), 239 (armwobble), 479 (wrist2), 508 (cubehold),
    monotone over 6000 iterations, entropy 18.5 -> 98.7.

    MECHANISM. Two forces act on `std_param`. The surrogate pushes sigma DOWN,
    but only in proportion to the CORRELATION between "how far did I deviate
    from mu" and "did it turn out better" — and for this task that correlation
    is ~0, because the finger action is coupled and clipped, so a large sampled
    action just means "fully closed", and fully closed holds the cube. The
    entropy bonus pushes sigma UP at dH/dsigma = 1/sigma, which decays — but
    ADAM IS SCALE-INVARIANT: for a consistently-signed gradient the update is
    ~lr regardless of magnitude, so the decay brakes nothing. 20 updates/iter x
    6000 iters = 120,000 updates at mean lr 3.886e-3 predicts a drift of 466;
    the observed sigma was 478.8.

    Two loops made it terminal: past the action clip the executed action is
    sign(noise), which decorrelates A from mu FURTHER; and KL ~ dmu^2/2sigma^2,
    so a growing sigma shrinks KL, the adaptive scheduler sees KL << desired_kl
    and RAISES lr (grasp ran at 3.9e-3 mean vs balance 3.8e-4, hitting the 1e-2
    clamp). The trust-region controller was flooring the throttle because the
    policy had become too noisy to change.

    WHY 0.0 AND NOT A SMALLER COEFFICIENT: under Adam the drift RATE is set by
    lr, not by the coefficient. What the coefficient changes is whether entropy
    still dominates the SIGN of std_param's gradient. Halving it barely helps;
    0.0 removes the upward force outright and leaves sigma to the surrogate.

    Exploration still exists: init_noise_std 1.0 (action clip is +-1), and sigma
    is free to move — it is simply no longer PAID to grow. If sigma now FALLS,
    the reward discriminates between actions; if it wanders, it does not, and
    the reward is the thing to fix. Either outcome is the diagnostic.
    """

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,          # <- the axis
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
