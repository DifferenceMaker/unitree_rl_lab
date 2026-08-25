# FastSAC runner for IsaacLab ManagerBasedRLEnv — ported from Amazon FAR's
# holosoma (Apache-2.0) fast_sac_agent.py. Update math VERBATIM; the holosoma
# framework plumbing (BaseAlgo/config registry/BaseTask/loguru/distributed/
# CNN encoder) is replaced by this lean single-GPU adapter.
#
# v1 KNOWN DELTAS vs the paper recipe (documented, revisit after the pilot):
#   * no symmetry augmentation (holosoma SymmetryUtils is framework-specific;
#     our mdp/symmetry.py can be adapted later — the paper reports symmetry
#     "helpful for faster convergence", not load-bearing)
#   * truncation bootstrapping uses the post-reset obs (their own fallback
#     path when final_observations are unavailable — IsaacLab resets in-step)
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch import optim
from torch.amp import GradScaler, autocast

from .networks import Actor, Critic
from .buffer import SimpleReplayBuffer, EmpiricalNormalization


@dataclass
class FastSACConfig:
    # verbatim defaults from holosoma config_values/algo.py `fast_sac`
    num_learning_iterations: int = 50000
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    buffer_size: int = 1024
    num_steps: int = 1
    gamma: float = 0.97
    tau: float = 0.125
    batch_size: int = 8192
    learning_starts: int = 10
    policy_frequency: int = 4
    num_updates: int = 8
    target_entropy_ratio: float = 0.0
    num_atoms: int = 101
    v_min: float = -20.0
    v_max: float = 20.0
    critic_hidden_dim: int = 768
    actor_hidden_dim: int = 512
    alpha_init: float = 0.001
    use_autotune: bool = True
    use_tanh: bool = True
    log_std_max: float = 0.0
    log_std_min: float = -5.0
    compile: bool = True
    obs_normalization: bool = True
    use_layer_norm: bool = True
    num_q_networks: int = 2
    max_grad_norm: float = 0.0
    amp: bool = True
    amp_dtype: str = "bf16"
    weight_decay: float = 0.001
    save_interval: int = 1000
    logging_interval: int = 100


def compute_action_boundaries(env) -> torch.Tensor:
    """Per-action tanh scaling: max range from default to either joint limit,
    divided by the action term's scale — action=0 lands on the default pose,
    action=+-1 reaches the furthest limit (holosoma _compute_action_boundaries
    re-derived for IsaacLab's JointPositionAction)."""
    term = None
    for name in env.action_manager.active_terms:
        t = env.action_manager.get_term(name)
        if hasattr(t, "_joint_ids") and hasattr(t, "_offset"):
            term = t
            break
    if term is None:
        raise RuntimeError("no JointPositionAction-like term found for boundaries")
    asset = env.scene["robot"]
    jids = term._joint_ids                                       # list or slice(None)
    limits = asset.data.soft_joint_pos_limits[0, jids]           # (n_act, 2)
    # the term's own offset IS the default reference (use_default_offset)
    offset = term._offset
    if isinstance(offset, torch.Tensor):
        default = offset[0]
    else:
        default = torch.full((limits.shape[0],), float(offset), device=env.device)
    scale = term._scale                                          # float or (n_env, n_act)
    if isinstance(scale, torch.Tensor):
        scale = scale[0]
    range_lo = (limits[:, 0] - default).abs()
    range_hi = (limits[:, 1] - default).abs()
    max_range = torch.maximum(range_lo, range_hi)
    return (max_range / scale).to(env.device)


class FastSACRunner:
    def __init__(self, env, cfg: FastSACConfig, log_dir: str, writer=None):
        """env: the RAW IsaacLab ManagerBasedRLEnv (not the rsl_rl wrapper)."""
        self.env = env
        self.config = cfg
        self.log_dir = log_dir
        self.writer = writer
        self.device = env.device
        self.global_step = 0

        args = cfg
        device = self.device
        self.num_envs = env.num_envs

        # obs dims from the manager groups: actor = "policy", critic = "critic"
        dims = {g: int(torch.tensor(env.observation_manager.group_obs_dim[g]).prod())
                for g in env.observation_manager.group_obs_dim}
        self.actor_obs_dim = dims["policy"]
        self.critic_obs_dim = dims.get("critic", dims["policy"])
        n_act = env.action_manager.total_action_dim
        self.n_act = n_act

        self.scaler = GradScaler(enabled=args.amp)
        if args.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=self.actor_obs_dim, device=device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=self.critic_obs_dim, device=device)
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.critic_obs_normalizer = torch.nn.Identity()

        action_scale = compute_action_boundaries(env) if args.use_tanh else torch.ones(n_act, device=device)
        print(f"[fastsac] obs {self.actor_obs_dim}/{self.critic_obs_dim} act {n_act} "
              f"| action_scale range [{action_scale.min():.2f}, {action_scale.max():.2f}]")

        self.actor = Actor(n_obs=self.actor_obs_dim, n_act=n_act, device=device,
                           hidden_dim=args.actor_hidden_dim, log_std_max=args.log_std_max,
                           log_std_min=args.log_std_min, use_tanh=args.use_tanh,
                           use_layer_norm=args.use_layer_norm, action_scale=action_scale)
        self.qnet = Critic(n_obs=self.critic_obs_dim, n_act=n_act, num_atoms=args.num_atoms,
                           v_min=args.v_min, v_max=args.v_max, hidden_dim=args.critic_hidden_dim,
                           device=device, use_layer_norm=args.use_layer_norm,
                           num_q_networks=args.num_q_networks)
        self.qnet_target = Critic(n_obs=self.critic_obs_dim, n_act=n_act, num_atoms=args.num_atoms,
                                  v_min=args.v_min, v_max=args.v_max, hidden_dim=args.critic_hidden_dim,
                                  device=device, use_layer_norm=args.use_layer_norm,
                                  num_q_networks=args.num_q_networks)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.log_alpha = torch.tensor([torch.log(torch.tensor(args.alpha_init)).item()],
                                      requires_grad=True, device=device)
        self.policy = self.actor.explore

        self.q_optimizer = optim.AdamW(list(self.qnet.parameters()), lr=args.critic_learning_rate,
                                       weight_decay=args.weight_decay, fused=True, betas=(0.9, 0.95))
        self.actor_optimizer = optim.AdamW(list(self.actor.parameters()), lr=args.actor_learning_rate,
                                           weight_decay=args.weight_decay, fused=True, betas=(0.9, 0.95))
        self.target_entropy = -n_act * args.target_entropy_ratio
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=args.alpha_learning_rate,
                                           fused=True, betas=(0.9, 0.95))

        self.rb = SimpleReplayBuffer(n_env=self.num_envs, buffer_size=args.buffer_size,
                                     n_obs=self.actor_obs_dim, n_act=n_act,
                                     n_critic_obs=self.critic_obs_dim, n_steps=args.num_steps,
                                     gamma=args.gamma, device=device)

    # ── env adapter ─────────────────────────────────────────────────────────
    def _split_obs(self, obs_dict):
        actor = obs_dict["policy"].reshape(self.num_envs, -1)
        critic = obs_dict.get("critic", obs_dict["policy"]).reshape(self.num_envs, -1)
        return actor, critic

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    # ── update math (verbatim port) ─────────────────────────────────────────
    def _update_main(self, data):
        args = self.config
        scaler, actor = self.scaler, self.actor
        qnet, qnet_target = self.qnet, self.qnet_target
        q_optimizer, alpha_optimizer = self.q_optimizer, self.alpha_optimizer

        with self._maybe_amp():
            next_observations = data["next"]["observations"]
            critic_observations = data["critic_observations"]
            next_critic_observations = data["next"]["critic_observations"]
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()

            with torch.no_grad():
                next_state_actions, next_state_log_probs = actor.get_actions_and_log_probs(next_observations)
                discount = args.gamma ** data["next"]["effective_n_steps"]
                target_distributions = qnet_target.projection(
                    next_critic_observations, next_state_actions,
                    rewards - discount * bootstrap * self.log_alpha.exp() * next_state_log_probs,
                    bootstrap, discount)
                target_values = qnet_target.get_value(target_distributions)
                target_value_max = target_values.max()
                target_value_min = target_values.min()

            q_outputs = qnet(critic_observations, actions)
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)
        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=args.max_grad_norm)
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(q_optimizer)
        scaler.update()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if args.use_autotune:
            alpha_optimizer.zero_grad(set_to_none=True)
            with self._maybe_amp():
                alpha_loss = (-self.log_alpha.exp() * (next_state_log_probs.detach() + self.target_entropy)).mean()
            scaler.scale(alpha_loss).backward()
            scaler.unscale_(alpha_optimizer)
            scaler.step(alpha_optimizer)
            scaler.update()

        return (rewards.mean(), critic_grad_norm.detach(), qf_loss.detach(),
                target_value_max.detach(), target_value_min.detach(), alpha_loss.detach())

    def _update_pol(self, data):
        actor, qnet = self.actor, self.qnet
        actor_optimizer, scaler, args = self.actor_optimizer, self.scaler, self.config

        with self._maybe_amp():
            critic_observations = data["critic_observations"]
            actions, log_probs = actor.get_actions_and_log_probs(data["observations"])
            with torch.no_grad():
                _, _, log_std = actor(data["observations"])
                action_std = log_std.exp().mean()
                policy_entropy = -log_probs.mean()
            q_outputs = qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = qnet.get_value(q_probs)
            qf_value = q_values.mean(dim=0)
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.max_grad_norm)
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        scaler.step(actor_optimizer)
        scaler.update()
        return (actor_grad_norm.detach(), actor_loss.detach(), policy_entropy.detach(), action_std.detach())

    def _sample_and_prepare_batches(self, batch_size, num_updates, normalize_obs, normalize_critic_obs):
        large_data = self.rb.sample(batch_size * num_updates)
        samples_per_update = batch_size * self.num_envs
        large_data["observations"] = normalize_obs(large_data["observations"])
        large_data["next"]["observations"] = normalize_obs(large_data["next"]["observations"])
        large_data["critic_observations"] = normalize_critic_obs(large_data["critic_observations"])
        large_data["next"]["critic_observations"] = normalize_critic_obs(large_data["next"]["critic_observations"])
        batches = []
        for i in range(num_updates):
            s, e = i * samples_per_update, (i + 1) * samples_per_update
            batches.append({
                "observations": large_data["observations"][s:e],
                "actions": large_data["actions"][s:e],
                "critic_observations": large_data["critic_observations"][s:e],
                "next": {k: large_data["next"][k][s:e] for k in large_data["next"]},
            })
        return batches

    # ── the learn loop (verbatim port, IsaacLab step contract) ──────────────
    def learn(self):
        args = self.config
        device = self.device
        if args.compile:
            update_main = torch.compile(self._update_main)
            update_pol = torch.compile(self._update_pol)
            policy = torch.compile(self.policy)
            normalize_obs = torch.compile(self.obs_normalizer.forward)
            normalize_critic_obs = torch.compile(self.critic_obs_normalizer.forward)
        else:
            update_main, update_pol, policy = self._update_main, self._update_pol, self.policy
            normalize_obs = self.obs_normalizer.forward
            normalize_critic_obs = self.critic_obs_normalizer.forward

        obs_dict, _ = self.env.reset()
        obs, critic_obs = self._split_obs(obs_dict)
        dones = None
        policy_entropy = torch.tensor(0.0, device=device)
        action_std = torch.tensor(0.0, device=device)
        actor_loss = torch.tensor(0.0, device=device)
        actor_grad_norm = torch.tensor(0.0, device=device)
        ep_rew = torch.zeros(self.num_envs, device=device)
        ep_len = torch.zeros(self.num_envs, device=device)
        fin_rew, fin_len, fin_n = 0.0, 0.0, 0
        t_last, step_last = time.monotonic(), 0

        while self.global_step <= args.num_learning_iterations:
            with torch.no_grad(), self._maybe_amp():
                norm_obs = normalize_obs(obs, update=False)
                actions = policy(obs=norm_obs, dones=dones)

            next_obs_dict, rewards, terminated, truncated, _extras = self.env.step(actions.float())
            next_obs, next_critic_obs = self._split_obs(next_obs_dict)
            dones = (terminated | truncated).to(device)
            truncations = truncated.to(device)

            ep_rew += rewards
            ep_len += 1
            if dones.any():
                idx = dones.nonzero(as_tuple=False).squeeze(-1)
                fin_rew += ep_rew[idx].sum().item()
                fin_len += ep_len[idx].sum().item()
                fin_n += len(idx)
                ep_rew[idx] = 0.0
                ep_len[idx] = 0.0

            # truncation bootstrapping: post-reset obs stand in for final obs
            # (holosoma's own fallback when final_observations are absent)
            transition = {
                "observations": obs,
                "actions": actions.float(),
                "critic_observations": critic_obs,
                "next": {
                    "observations": next_obs,
                    "rewards": rewards.float().to(device),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                    "critic_observations": next_critic_obs,
                },
            }
            obs, critic_obs = next_obs, next_critic_obs
            self.rb.extend(transition)

            batch_size = max(args.batch_size // self.num_envs, 1)
            if self.global_step > args.learning_starts:
                batches = self._sample_and_prepare_batches(
                    batch_size, args.num_updates, normalize_obs, normalize_critic_obs)
                for i, data in enumerate(batches):
                    (buffer_rewards, critic_grad_norm, qf_loss, qf_max, qf_min,
                     alpha_loss) = update_main(data)
                    if args.num_updates > 1:
                        if i % args.policy_frequency == 1:
                            actor_grad_norm, actor_loss, policy_entropy, action_std = update_pol(data)
                    elif self.global_step % args.policy_frequency == 0:
                        actor_grad_norm, actor_loss, policy_entropy, action_std = update_pol(data)
                    with torch.no_grad():
                        src_ps = [p.data for p in self.qnet.parameters()]
                        tgt_ps = [p.data for p in self.qnet_target.parameters()]
                        torch._foreach_mul_(tgt_ps, 1.0 - args.tau)
                        torch._foreach_add_(tgt_ps, src_ps, alpha=args.tau)

                # NOTE (verbatim holosoma behavior): the obs normalizers update
                # from the replayed batches inside _sample_and_prepare_batches
                # (module in train mode, update defaults True); rollout-side
                # normalization is always update=False.

                if self.global_step % args.logging_interval == 0:
                    now = time.monotonic()
                    sps = (self.global_step - step_last) * self.num_envs / max(1e-6, now - t_last)
                    t_last, step_last = now, self.global_step
                    mr = fin_rew / max(1, fin_n)
                    ml = fin_len / max(1, fin_n)
                    fin_rew, fin_len, fin_n = 0.0, 0.0, 0
                    print(f"FastSAC step {self.global_step}/{args.num_learning_iterations} "
                          f"| {sps:,.0f} steps/s | mean_rew {mr:.2f} | mean_len {ml:.0f} "
                          f"| qf[{qf_min:.1f},{qf_max:.1f}] | alpha {self.log_alpha.exp().item():.4f} "
                          f"| std {action_std.item():.3f}", flush=True)
                    if self.writer is not None:
                        w = self.writer
                        w.add_scalar("Train/mean_reward", mr, self.global_step)
                        w.add_scalar("Train/mean_episode_length", ml, self.global_step)
                        w.add_scalar("Loss/qf_loss", qf_loss.item(), self.global_step)
                        w.add_scalar("Loss/actor_loss", actor_loss.item(), self.global_step)
                        w.add_scalar("Policy/mean_std", action_std.item(), self.global_step)
                        w.add_scalar("Policy/entropy", policy_entropy.item(), self.global_step)
                        w.add_scalar("Policy/alpha", self.log_alpha.exp().item(), self.global_step)
                        w.add_scalar("Perf/steps_per_s", sps, self.global_step)

                if args.save_interval > 0 and self.global_step > 0 and self.global_step % args.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{self.global_step}.pt"))
                    self.export_onnx(os.path.join(self.log_dir, f"model_{self.global_step}.onnx"))

            if self.global_step >= args.num_learning_iterations:
                break
            self.global_step += 1

        self.save(os.path.join(self.log_dir, f"model_{self.global_step}.pt"))
        self.export_onnx(os.path.join(self.log_dir, f"model_{self.global_step}.onnx"))

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "qnet_state_dict": self.qnet.state_dict(),
            "qnet_target_state_dict": self.qnet_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "obs_normalizer_state": self.obs_normalizer.state_dict()
                if hasattr(self.obs_normalizer, "state_dict") else None,
            "critic_obs_normalizer_state": self.critic_obs_normalizer.state_dict()
                if hasattr(self.critic_obs_normalizer, "state_dict") else None,
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
            "grad_scaler_state_dict": self.scaler.state_dict(),
            "args": asdict(self.config),
            "global_step": self.global_step,
        }, path)
        print(f"[fastsac] saved {path}", flush=True)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.qnet.load_state_dict(ckpt["qnet_state_dict"])
        self.qnet_target.load_state_dict(ckpt["qnet_target_state_dict"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        if ckpt.get("obs_normalizer_state") is not None:
            self.obs_normalizer.load_state_dict(ckpt["obs_normalizer_state"])
        if ckpt.get("critic_obs_normalizer_state") is not None:
            self.critic_obs_normalizer.load_state_dict(ckpt["critic_obs_normalizer_state"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(ckpt["q_optimizer_state_dict"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer_state_dict"])
        if "grad_scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["grad_scaler_state_dict"])
        self.global_step = ckpt["global_step"]
        print(f"[fastsac] loaded {path} @ step {self.global_step}", flush=True)

    def export_onnx(self, path):
        """Deterministic actor with the obs normalizer folded in — the same
        [1, obs] -> [1, act] contract our PPO exports use. NOTE the output is
        the FINAL scaled action (tanh(mean)*scale): deploy-side this is the
        raw-action replacement BEFORE the usual apply_action_transform."""
        actor = self.actor

        class _Deploy(torch.nn.Module):
            def __init__(self, actor, normalizer):
                super().__init__()
                self.actor = actor
                self.normalizer = normalizer

            def forward(self, obs):
                if not isinstance(self.normalizer, torch.nn.Identity):
                    obs = (obs - self.normalizer._mean) / (self.normalizer._std + self.normalizer.eps)
                action, _, _ = self.actor(obs)
                return action

        m = _Deploy(actor, self.obs_normalizer).eval()
        dummy = torch.zeros(1, self.actor_obs_dim, device=self.device)
        torch.onnx.export(m, dummy, path, input_names=["obs"], output_names=["actions"],
                          opset_version=17)
        print(f"[fastsac] exported {path}", flush=True)
