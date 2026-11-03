"""LM7J terms (2026-11-02): the gradual stop -- a slewed velocity command.

Operator 2026-11-02 (robot test of lm7i_combo_soft): "another thing which i see is the immediate stop when locomoting. IT counter-sways like
crazy. What i would like more is if it gradually stopped ... the more velocity it has, the bigger the window for stopping"; "Maybe we
implement this kind of deceleration for stopping in training too? instead of hard-stopping?" -> lm7j_stopramp.

  SlewVelocityCommand   CornerVelocityCommand whose OUTPUT (vel_command_b: what the observation, every reward, the metrics, the
                        handover hold and reanchor_on_stop read) moves toward the sampled target at a bounded rate. Per axis: an axis
                        moving TOWARD zero (|target| < |out|, or the target on the other side of zero) changes by at most
                        slew_dec * step_dt per step (through zero it first reaches 0); an axis moving away from zero by at most
                        slew_acc * step_dt (inf = the old instant step). So a stop from v0 takes T = |v0| / a: at a = 1.0 m/s^2,
                        0.8 m/s -> 0.8 s, 0.1 m/s -> 0.1 s. The standing gate (|cmd| < cmd_thr) is computed on the output, so the
                        standing terms (tight zero-command tracking, velocity / step prices, the position hold re-anchored at the stop)
                        switch on only at the end of the ramp -- an OBSERVABLE window (the policy sees the ramping command; Bible
                        rule 32), unlike a reward window on the velocity at the stop (the actor has no base_lin_vel).
                        Resets jump (out = target on the first compute after a reset); an external write to vel_command_b (the bank's
                        handover hold zeroes it after the compute) is honoured because each step slews from the CURRENT output.

  ArmGravityFFJointPositionAction  (lm7j_ffarms, 2026-11-02, gene A = plant parity) JointPositionAction whose apply_actions ALSO writes
                        the generalized gravity torque of the 14 arm DOFs as the joint EFFORT target every physics step. The robot's
                        ActionModule streams pinocchio g(q) for both arms to /BridgeModule/arm_tau_ff (AM_ARM_GRAV_FF on by default
                        since 2026-09-15, scale 1.0, the Bridge clips at 25 Nm per motor) and the firmware law is
                        tau = tau_ff + kp (q* - q) - kd dq; Isaac never had the tau_ff, so in sim the arms sag by g(q) / kp (kp 40:
                        5 Nm = 7.2 deg) and every policy learned THAT plant. Robot A/B 2026-11-02 (AM_ARM_GRAV_FF=0 vs 1): the
                        mismatch is the balance -> walk handover lean and the stance lean. Here: IdealPDActuator.compute adds
                        control_action.joint_efforts (= Articulation.joint_effort_target) to kp e + kd de before the effort clip, and
                        ManagerBasedEnv.step calls apply_action before every one of the decimation physics steps, so g(q) is
                        recomputed at the current pose at 200 Hz (the robot: 30 Hz). Source: root_physx_view.get_gravity_compensation_forces()
                        -- (count, max_dofs + 6) for a floating base (6 root columns FIRST), (count, max_dofs) for a fixed base; the
                        joint columns are found by shape at construction. Proof: scripts/tools/check_arm_gravity_ff.py (aspired-isaac-lab).

2026-11-03 (lm7j_stack / lm7j_stack_base / lm7j_legs on lm7j_combo @133393; operator's genes after the Exam v3b re-land: every row but
ffarms slides OUT after a walk, clear did not lift the feet, the stop recoils):
  slew_dec_ramp         curriculum: SlewVelocityCommand's deceleration limit ramped linearly between two absolute common_step_counter
                        ticks, (1.0, 1.0, 2.0) -> (0.5, 0.5, 1.0): a 0.8 m/s stop takes 0.8 s at the start and 1.6 s at the end.
  arm_motion_walking    L2 of the 8 upper-arm joint velocities (shoulder pitch / roll / yaw + elbow pitch, both arms) while walking
                        (|cmd| > cmd_thr, the observed command). The elbow_roll / wrist_pitch / wrist_yaw DOFs are EXCLUDED: the
                        explicit-PD wrist group is a plant limit cycle (elbow_roll +-9.5 rad/s at 19 Nm in every policy, the 09-08
                        armature bake, see wrist-actuator-chatter) -- pricing it would price the plant, not the sway.
  feet_width_hold_standing   exp(-(w - w0)^2 / std^2) at a zero command, w = the lateral foot separation in the base frame (left y -
                        right y), w0 = the separation at the spawn / the LAST STOP (events.capture_spawn_state / reanchor_on_stop now
                        carry env.spawn_feet_width). The post-walk slide-out priced from where the feet landed.
  feet_width_walking    exp(-(w - target)^2 / std^2) under a command: the walking width toward 0.30 m (the legs-in walk at 22-27 cm).
  feet_yaw_hold_standing   mean over the feet of exp(-(yaw_foot - yaw_base)^2 / std^2) at a zero command, std 0.1 rad: live at 3-9 deg
                        where feet_yaw_bonus (L1, sigma 3 deg) has already gone flat (lm7d's footyaw L2 was flat at 0).
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from .events import feet_lateral_sep_b, feet_yaw_rel_base

from .commands.velocity_command import CornerVelocityCommand, UniformLevelVelocityCommandCfg


def torso_roll_rate_l2_walking(env, command_name: str = "base_velocity", cmd_thr: float = 0.03,
                               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link")) -> torch.Tensor:
    """lm7j_roll (operator 2026-11-02: "Looking at the shoulder line, it goes left and right while walking. I need that shoulder line to be
    there"): the torso ROLL rate squared -- torso_link angular velocity about its own x axis (not pitch, not yaw) -- while walking
    (|command| >= cmd_thr: the observed command, Bible rule 32; the standing side already has torso_ang_vel / torso_still). An L2 price,
    no kernel: the gradient grows with the swing instead of going flat where the robot lives (the torso15 problem, rule 31)."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    q = asset.data.body_quat_w[:, ids, :].reshape(-1, 4)
    w = asset.data.body_ang_vel_w[:, ids, :].reshape(-1, 3)
    wx = quat_apply_inverse(q, w)[:, 0]
    walking = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) >= cmd_thr
    return torch.square(wx) * walking.float()


class SlewVelocityCommand(CornerVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._target = torch.zeros_like(self.vel_command_b)
        self._jump = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._dec = torch.tensor([float(x) for x in cfg.slew_dec], device=self.device)
        self._acc = torch.tensor([float(x) for x in cfg.slew_acc], device=self.device)
        self._watch, self._trace, self._traces_printed, self._ramping_steps = -1, [], 0, 0
        self._swatch, self._strace, self._straces_printed = -1, [], 0   # a START (acceleration) ramp, when slew_acc is finite
        print(f"[lm7j] slew command: dec {tuple(cfg.slew_dec)} (m/s^2, m/s^2, rad/s^2), acc {tuple(cfg.slew_acc)}, "
              f"step_dt {env.step_dt:.3f} s", flush=True)

    def reset(self, env_ids: Sequence[int] | None = None):
        extras = super().reset(env_ids)
        self._jump[slice(None) if env_ids is None else env_ids] = True
        return extras

    def set_dec(self, dec) -> None:
        """The deceleration limit per axis (slew_dec_ramp curriculum, 2026-11-03); the cfg mirrors it for a reader of the term."""
        self._dec[:] = torch.tensor([float(x) for x in dec], device=self.device)
        self.cfg.slew_dec = tuple(float(x) for x in dec)

    def _resample_command(self, env_ids: Sequence[int]):
        out_prev = self.vel_command_b[env_ids].clone()
        super()._resample_command(env_ids)                  # the new TARGET (uniform / corner / slow / spin draw)
        self._target[env_ids] = self.vel_command_b[env_ids]
        self.vel_command_b[env_ids] = out_prev               # the output keeps its value; _update_command slews it

    def _update_command(self):
        out = self.vel_command_b.clone()                    # the CURRENT output, incl. an external write (the handover hold)
        self.vel_command_b[:] = self._target
        super()._update_command()                           # heading envs: wz from the heading error; standing envs: zero
        tgt = self.vel_command_b.clone()                    # the final target of this step
        dt = float(self._env.step_dt)
        toward_zero = (tgt.abs() < out.abs()) | (tgt * out < 0)
        goal = torch.where(tgt * out < 0, torch.zeros_like(tgt), tgt)    # through zero: reach 0 first, then accelerate
        lim = torch.where(toward_zero, self._dec, self._acc) * dt         # (N, 3); inf = instant
        new = torch.where(torch.isinf(lim), goal, out + torch.maximum(torch.minimum(goal - out, lim), -lim))
        new[self._jump] = tgt[self._jump]
        self._jump[:] = False
        self.vel_command_b[:] = new
        self._proof(tgt, new)

    def _proof(self, tgt, new):
        """Proof of life in the training log (JOB_VALIDITY 13): one stop traced from its start to the end of the ramp."""
        ramping = (tgt - new).abs().amax(dim=1) > 1e-6
        self._ramping_steps += int(ramping.sum())
        if self._env.common_step_counter % 2400 == 1:
            print(f"[lm7j] slew: {int(ramping.sum())} envs ramping now, {self._ramping_steps} env-steps ramped so far", flush=True)
        if self._straces_printed < 3 and not bool(torch.isinf(self._acc).all()):
            if self._swatch < 0:
                cand = (ramping & (new[:, :2].norm(dim=1) < 0.05) & (tgt[:, :2].norm(dim=1) > 0.3)).nonzero().flatten()
                if len(cand):
                    self._swatch, self._strace = int(cand[0]), []
            if self._swatch >= 0:
                i = self._swatch
                o = new[i].tolist()
                self._strace.append((round(o[0], 3), round(o[1], 3), round(o[2], 3)))
                if not bool(ramping[i]) or len(self._strace) > 200:
                    n = len(self._strace)
                    print(f"[lm7j] start trace env {i} ({n} steps = {n * self._env.step_dt:.2f} s, target {[round(x, 3) for x in tgt[i].tolist()]}): "
                          + " ".join(f"{t[0]:+.2f}/{t[1]:+.2f}/{t[2]:+.2f}" for t in self._strace[::max(1, n // 12)]), flush=True)
                    self._straces_printed += 1
                    self._swatch = -1
        if self._traces_printed >= 3:
            return
        if self._watch < 0:
            cand = (ramping & (tgt.abs().amax(dim=1) < 0.03) & (new[:, :2].norm(dim=1) > 0.3)).nonzero().flatten()
            if len(cand):
                self._watch, self._trace = int(cand[0]), []
        if self._watch >= 0:
            i = self._watch
            o = new[i].tolist()
            self._trace.append((round(o[0], 3), round(o[1], 3), round(o[2], 3), bool(new[i].norm() < 0.03)))
            if not bool(ramping[i]) or len(self._trace) > 200:
                n = len(self._trace)
                print(f"[lm7j] stop trace env {i} ({n} steps = {n * self._env.step_dt:.2f} s, target {tgt[i].tolist()}): "
                      + " ".join(f"{t[0]:+.2f}/{t[1]:+.2f}/{t[2]:+.2f}{'*' if t[3] else ''}" for t in self._trace[::max(1, n // 12)])
                      + "  (* = standing gate |cmd| < .03 open)", flush=True)
                self._traces_printed += 1
                self._watch = -1


@configclass
class SlewVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = SlewVelocityCommand
    slew_dec: tuple = (1.0, 1.0, 2.0)
    """Bounded rate toward zero per axis (vx m/s^2, vy m/s^2, wz rad/s^2)."""
    slew_acc: tuple = (math.inf, math.inf, math.inf)
    """Bounded rate away from zero (inf = instant, the pre-lm7j behaviour)."""


def slew_command_cfg(c, dec=(1.0, 1.0, 2.0), acc=(math.inf, math.inf, math.inf)) -> SlewVelocityCommandCfg:
    """A SlewVelocityCommandCfg carrying every field of the existing (UniformLevel) velocity command cfg `c`."""
    kw = {f.name: getattr(c, f.name) for f in dataclasses.fields(c) if f.name != "class_type"}
    n = SlewVelocityCommandCfg(**kw)
    n.slew_dec, n.slew_acc = tuple(dec), tuple(acc)
    return n


# ---------------------------------------------------------------------------------------------------------------------------------
# gene A (lm7j_ffarms): arm gravity feed-forward in sim = the robot's plant
# ---------------------------------------------------------------------------------------------------------------------------------
ARM_FF_JOINTS = (".*_shoulder_pitch.*", ".*_shoulder_roll.*", ".*_shoulder_yaw.*", ".*_elbow.*", ".*_wrist.*")   # the 14 arm DOFs = joint_deviation_arms' set
ARM_FF_CLIP_NM = 25.0   # Bridge joint_set.py clips the streamed tau_ff at 25 Nm per motor (ActionModule/main/robot/gravity_ff.py)


class ArmGravityFFJointPositionAction(JointPositionAction):
    """JointPositionAction + the generalized gravity torque of the arm DOFs as a feed-forward effort (see the module docstring)."""

    cfg: "ArmGravityFFJointPositionActionCfg"

    def __init__(self, cfg: "ArmGravityFFJointPositionActionCfg", env):
        super().__init__(cfg, env)
        ids, names = self._asset.find_joints(list(cfg.arm_gravity_ff_joints))
        self._ff_ids = list(ids)
        self._ff_ids_t = torch.tensor(self._ff_ids, dtype=torch.long, device=self.device)
        self._ff_col0 = None                                  # set on the first call from the API's shape
        self._ff_steps, self._ff_sum, self._ff_max = 0, 0.0, 0.0
        print(f"[lm7j] arm gravity ff: {'ON' if cfg.arm_gravity_ff else 'OFF'}, {len(self._ff_ids)} joints {names}, "
              f"clip {cfg.arm_gravity_ff_clip} Nm, scale {cfg.arm_gravity_ff_scale}", flush=True)

    def _arm_gravity(self) -> torch.Tensor:
        g = self._asset.root_physx_view.get_gravity_compensation_forces()
        if self._ff_col0 is None:
            n = self._asset.num_joints
            if g.shape[1] == n:
                self._ff_col0 = 0
            elif g.shape[1] == n + 6:
                self._ff_col0 = 6                             # floating base: the 6 root wrench columns come first
            else:
                raise RuntimeError(f"[lm7j] gravity compensation forces shape {tuple(g.shape)} vs {n} joints")
            print(f"[lm7j] arm gravity ff: get_gravity_compensation_forces shape {tuple(g.shape)} for {n} joints -> joint columns start at {self._ff_col0}", flush=True)
        return g[:, self._ff_col0 + self._ff_ids_t]

    def apply_actions(self):
        super().apply_actions()
        if not self.cfg.arm_gravity_ff:
            return
        tau = self._arm_gravity() * self.cfg.arm_gravity_ff_scale
        if self.cfg.arm_gravity_ff_clip > 0:
            tau = tau.clamp(-self.cfg.arm_gravity_ff_clip, self.cfg.arm_gravity_ff_clip)
        self._asset.set_joint_effort_target(tau, joint_ids=self._ff_ids)
        self._proof(tau)

    def _proof(self, tau):
        """Proof of life in the training log (JOB_VALIDITY 13): the mean / max |tau_ff| over the arm DOFs, every 2400 env steps."""
        self._ff_steps += 1
        a = tau.abs()
        self._ff_sum += float(a.mean()); self._ff_max = max(self._ff_max, float(a.max()))
        if self._env.common_step_counter % 2400 == 1 and self._ff_steps % int(getattr(self._env.cfg, "decimation", 1)) == 1:
            per = a.mean(dim=0)
            print(f"[lm7j] arm gravity ff: mean |tau_ff| {self._ff_sum / self._ff_steps:.3f} Nm (running), max {self._ff_max:.2f} Nm; "
                  f"now per joint {[round(x, 2) for x in per.tolist()]}", flush=True)


@configclass
class ArmGravityFFJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ArmGravityFFJointPositionAction
    arm_gravity_ff: bool = True
    """Add the generalized gravity torque of `arm_gravity_ff_joints` as the joint effort target every physics step."""
    arm_gravity_ff_joints: tuple = ARM_FF_JOINTS
    arm_gravity_ff_clip: float = ARM_FF_CLIP_NM
    """Per-joint clip of the feed-forward (Nm, the Bridge's 25); <= 0 = no clip. The actuator's own effort limit clips the sum after."""
    arm_gravity_ff_scale: float = 1.0
    """AM_ARM_GRAV_FF_SCALE (1.0 since 2026-09-16: the hand790 URDF carries the Inspire mass)."""


def arm_gravity_ff_action_cfg(a) -> ArmGravityFFJointPositionActionCfg:
    """An ArmGravityFFJointPositionActionCfg carrying every field of the existing JointPositionActionCfg `a` (scale, offset, clip, joints)."""
    kw = {f.name: getattr(a, f.name) for f in dataclasses.fields(a) if f.name != "class_type"}
    return ArmGravityFFJointPositionActionCfg(**kw)


# ---------------------------------------------------------------------------------------------------------------------------------
# 2026-11-03 rows: lm7j_stack / lm7j_stack_base / lm7j_legs (see the module docstring)
# ---------------------------------------------------------------------------------------------------------------------------------
ARM_MOTION_JOINTS = (".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_pitch_joint")   # 8 DOFs; NOT elbow_roll / wrist_* (the chatter group)


def slew_dec_ramp(env, env_ids, command_name: str = "base_velocity", dec_from: tuple = (1.0, 1.0, 2.0), dec_to: tuple = (0.5, 0.5, 1.0),
                  start_step: int = 0, ramp_steps: int = 48000) -> torch.Tensor:
    """Curriculum: the SlewVelocityCommand's deceleration limit, linear from dec_from (tick <= start_step) to dec_to (tick >= start_step +
    ramp_steps) on ABSOLUTE env.common_step_counter ticks (curriculums.reward_param_ramp's convention: a --resume_curriculum row anchors
    the ramp at (parent_it + N) x 24). The stop window T = |v0| / a_dec grows with the ramp: 0.8 m/s -> 0.8 s at the start, 1.6 s at
    the end. Returns the vx limit for the Curriculum/<name> log; a proof line at the first call, per 2400 ticks until the end, at the end."""
    term = env.command_manager.get_term(command_name)
    assert isinstance(term, SlewVelocityCommand), f"[slew_dec_ramp] {command_name} is {type(term).__name__}, not SlewVelocityCommand"
    step = int(env.common_step_counter)
    u = 0.0 if step <= int(start_step) else min(1.0, (step - int(start_step)) / float(max(1, int(ramp_steps))))
    dec = [float(a) + u * (float(b) - float(a)) for a, b in zip(dec_from, dec_to)]
    term.set_dec(dec)
    st = getattr(env, "_slew_dec_ramp", None)
    if st is None or (not st["done"] and (step - st["step"] >= 2400 or u >= 1.0)):   # every 100 it until the end; the resume jump re-prints
        print(f"[lm7j] slew dec ramp: {tuple(round(x, 4) for x in dec)} (u {u:.3f}; tick {step}, ramp {int(start_step)} -> {int(start_step) + int(ramp_steps)} "
              f"= it {int(start_step) // 24} -> {(int(start_step) + int(ramp_steps)) // 24}); 0.8 m/s stops in {0.8 / dec[0]:.2f} s", flush=True)
        env._slew_dec_ramp = {"step": step, "done": u >= 1.0}
    return torch.tensor(dec[0], device=env.device)


def _walking(env, command_name: str, cmd_thr: float) -> torch.Tensor:
    return torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > cmd_thr


def arm_motion_walking(env, asset_cfg: SceneEntityCfg, command_name: str = "base_velocity", cmd_thr: float = 0.05) -> torch.Tensor:
    """Sum over asset_cfg.joint_ids (the 8 upper-arm DOFs) of joint_vel^2 while |cmd| > cmd_thr (the observed command, Bible rule 32).
    An L2 price (no kernel, no flat region): at weight -0.25 one joint at 1 rad/s costs 0.25 / step, three joints 0.75 / step. NEGATIVE
    weight. The standing side is already priced by joint_vel_reg (-0.01, all joints) and joint_deviation_arms (L1 to default, -6)."""
    asset = env.scene[asset_cfg.name]
    v = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(v * v, dim=1) * _walking(env, command_name, cmd_thr).float()


def feet_width_hold_standing(env, asset_cfg: SceneEntityCfg, std: float = 0.03, command_name: str = "base_velocity", cmd_thr: float = 0.03) -> torch.Tensor:
    """exp(-(w - w0)^2 / std^2) at a zero command: w = the lateral foot separation in the base frame, w0 = env.spawn_feet_width (the
    separation at the spawn, re-captured at EVERY stop by events.reanchor_on_stop). std 0.03: 1 cm of slide pays 0.90, 3 cm 0.37,
    6 cm 0.02; the gradient peaks at 2.1 cm = where the post-walk slide starts (the walls feet_too_near / _far catch the rest).
    POSITIVE weight; 0 until the anchor exists."""
    if not hasattr(env, "spawn_feet_width"):
        return torch.zeros(env.num_envs, device=env.device)
    w = feet_lateral_sep_b(env, asset_cfg)
    e = w - env.spawn_feet_width
    return torch.exp(-e * e / (std * std)) * (~_walking(env, command_name, cmd_thr)).float()


def feet_width_walking(env, asset_cfg: SceneEntityCfg, target: float = 0.30, std: float = 0.06, command_name: str = "base_velocity",
                       cmd_thr: float = 0.03) -> torch.Tensor:
    """exp(-(w - target)^2 / std^2) under a command: the lateral separation toward target while walking (the parent walks at 22-27 cm
    and the stand is 32: the narrow walk is where the legs go inward). std 0.06: 24 cm pays 0.37, 27 cm 0.78. POSITIVE weight."""
    w = feet_lateral_sep_b(env, asset_cfg)
    e = w - target
    return torch.exp(-e * e / (std * std)) * _walking(env, command_name, cmd_thr).float()


def feet_yaw_hold_standing(env, asset_cfg: SceneEntityCfg, std: float = 0.1, command_name: str = "base_velocity", cmd_thr: float = 0.03) -> torch.Tensor:
    """Mean over the feet of exp(-(yaw_foot - yaw_base)^2 / std^2) at a zero command (ceiling 1.0). std 0.1 rad: a foot 4 deg out
    pays 0.61, 5.7 deg 0.37, 10 deg 0.05 -- live where feet_yaw_bonus (L1, sigma 3 deg: 5.7 deg pays 0.15, 10 deg 0.04 for ONE foot)
    has gone flat. POSITIVE weight."""
    y = feet_yaw_rel_base(env, asset_cfg)
    return torch.exp(-y * y / (std * std)).mean(dim=1) * (~_walking(env, command_name, cmd_thr)).float()
