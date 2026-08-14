# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


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


@configclass
class GraspNoEntropyFixedLRPPORunnerCfg(GraspNoEntropyPPORunnerCfg):
    """gr5d: entropy_coef 0.0 AND a FIXED learning rate (3e-4).

    gr5c proved the adaptive-KL scheduler and sigma form a feedback loop with no
    stable operating point at EITHER end:
      * sigma large (gr5b): KL ~ dmu^2/2sigma^2 shrinks -> scheduler sees
        KL << desired_kl -> RAISES lr toward the 1e-2 clamp -> sigma drifts
        faster (Adam moves std_param ~lr per update). The runaway.
      * sigma small (gr5c_sigfix): same formula grows KL -> scheduler CUTS lr
        to the 1e-5 floor by ~iter 1500 -> learning freezes for the remaining
        70% of the budget (sigma 0.075 -> 0.056 and reward flat 119 -> 119
        from iter 1500 to 4999).
    A fixed lr severs the loop instead of hunting a magic entropy_coef between
    0.0 and 0.01. 3e-4 = the band healthy balance runs settle into under the
    adaptive scheduler (measured mean 3.795e-4), and the band this task itself
    passed through while it was still learning (iter 400-700: 5.9e-4 -> 2.6e-4).
    """

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,      # <- fixed, no schedule
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,           # unused under schedule="fixed"
        max_grad_norm=1.0,
    )


@configclass
class RslRlLcpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO + Lipschitz gradient penalty (paper #14) — see agents/lcp_ppo.py.
    Extra fields ride into LCPPPO.__init__ as kwargs (construct_algorithm
    passes the whole algorithm cfg dict through)."""

    class_name: str = "unitree_rl_lab.tasks.locomotion.agents.lcp_ppo:LCPPPO"
    lcp_coef: float = 0.05
    lcp_num_steps: int = 4
    lcp_batch_size: int = 4096


@configclass
class LM4PPORunnerCfg(WalkPPORunnerCfg):
    """lm4: WalkPPORunnerCfg + left-right mirror loss (paper #226).

    Mirror loss only, NO data augmentation: augmented (obs, action) pairs were
    never sampled from the current policy, which strains PPO's importance
    ratios; the soft mirror-consistency loss on the actor mean has no such
    off-policy cost. The mirror map is built from the LIVE env at first use
    (mdp/symmetry.py) and hard-fails on any obs contract it does not know.
    """

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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=1.0,
            data_augmentation_func="unitree_rl_lab.tasks.locomotion.mdp.symmetry:mirror_h1_2_walk",
        ),
    )


@configclass
class LM4LcpPPORunnerCfg(LM4PPORunnerCfg):
    """lm4_lcp: the same symmetry-enabled runner with LCPPPO — action_rate is
    deleted in the env cfg (RobotEnvCfgLM4LCP) and replaced by the gradient
    penalty in the loss. Verifiable post-hoc in the harvested agent.yaml
    (class_name + lcp_* fields), the project's ground truth."""

    algorithm = RslRlLcpPpoAlgorithmCfg(
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=1.0,
            data_augmentation_func="unitree_rl_lab.tasks.locomotion.mdp.symmetry:mirror_h1_2_walk",
        ),
    )


@configclass
class LM4BMirror01PPORunnerCfg(LM4PPORunnerCfg):
    """lm4b_mirror01: the lm4 runner with mirror_loss_coeff 1.0 -> 0.1.

    lm4's measured pathology: mirror loss ~0.008 x coeff 1.0 vs surrogate
    ~0.003 — the symmetry constraint outweighed the task gradient 3-4x, sigma
    never annealed (0.96-0.99 for 16k iters vs cleargate 0.99->0.77), survival
    plateaued at ~24 s. 0.1 keeps the symmetry prior (paper #226) at a dose
    that cannot dominate. A task-id-bound cfg, verifiable post-hoc in the
    harvested agent.yaml (the gr5c silent-override lesson)."""

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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
            data_augmentation_func="unitree_rl_lab.tasks.locomotion.mdp.symmetry:mirror_h1_2_walk",
        ),
    )


@configclass
class LM4BLcpMirror01PPORunnerCfg(LM4PPORunnerCfg):
    """lm4b_lcp: LCP retest with the sigma anchor KEPT (operator, 2026-08-14).

    lm4_lcp (attempt 1) NaN'd at iter 641: deleting action_rate removed the
    only term where a sampled deviation reliably costs reward — the
    deviation-outcome correlation collapsed, entropy 0.01 ground sigma up
    (1.0 -> 1.67) under Adam scale-invariance, the KL-adaptive lr amplified,
    std went NaN. The LCP penalty regularizes mu w.r.t. obs and has NO
    gradient path to std_param — it cannot inherit the anchor job.

    This retest keeps a SMALL action_rate (-0.1, job-side set_weight: anchor
    intact, bribe mostly gone) and runs the gradient penalty on top of
    mirror 0.1 — at mirror 1.0 the run would inherit lm4's suspected
    mirror-dominance stall and the LCP question would be unanswerable.
    Sibling axis vs lm4b_mirror01: {action_rate -0.6 -> -0.1 + LCPPPO}.
    Sigma trajectory is the first read: bounded = anchor dose sufficient."""

    algorithm = RslRlLcpPpoAlgorithmCfg(
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=0.1,
            data_augmentation_func="unitree_rl_lab.tasks.locomotion.mdp.symmetry:mirror_h1_2_walk",
        ),
    )
