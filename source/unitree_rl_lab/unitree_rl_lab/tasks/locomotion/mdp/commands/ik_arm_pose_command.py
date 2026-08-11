"""IK-resolved arm pose command — the "IK envelope" (p11).

Replaces the joint-angle envelope (UniformArmPoseCommand) with what the robot
actually does at deployment: the ActionModule solves MoveIt IK for Cartesian
hand targets and streams the resulting joint vectors. Training now mirrors
that pipeline:

  - Every resample (2-10 s, mid-episode — via the standard CommandTerm
    resampling machinery), each hand independently draws a Cartesian target
    in the TORSO frame (same frame the deployment IK solves in). Asymmetric
    L/R targets are the norm, not a curriculum level.
  - Each step, a batched damped-least-squares differential IK controller
    (isaaclab.controllers.DifferentialIKController — same pattern as Isaac
    Lab's own test_differential_ik.py) pulls the 14 arm joint targets toward
    the hand targets, rate-limited to max_joint_speed. Once a hand converges
    (or settle_time expires — the analog of the deployment's partial-reach
    bisection), its side of the command freezes: piecewise-constant held
    pose, exactly like a streamed IK solution.
  - With probability default_pose_prob a resample commands the DEFAULT pose
    instead (arms down) — the "peace-time" anchor: quiet standing must stay
    in-distribution.

The policy observation is unchanged: the 14-dim absolute arm joint targets
(same joint order as UniformArmPoseCommand) — warmstart compatible.

Redundancy note: DLS seeded from the current pose resolves the 7-DOF arm's
elbow null-space the same local way MoveIt's seeded KDL solver does, so the
elbow "style" seen in training matches deployment.

debug_vis=True shows one sphere per hand target (red = left, blue = right).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class IKArmPoseCommand(CommandTerm):
    """Cartesian hand targets resolved to joint targets by differential IK.

    See module docstring. The command tensor is the (num_envs, 14) absolute
    arm joint targets, identical in shape/order/semantics to
    UniformArmPoseCommand's, so policies transfer between the two.
    """

    cfg: "IKArmPoseCommandCfg"

    def __init__(self, cfg: "IKArmPoseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # 14 arm joints in asset order — same regex as UniformArmPoseCommand
        # so the observation layout (and thus warmstarts) are unchanged.
        self.all_arm_joint_ids, self.all_arm_joint_names = self.robot.find_joints(cfg.all_arm_joint_names)
        self.num_arm_joints = len(self.all_arm_joint_ids)

        # Per-arm joint ids + their column index inside the 14-dim command.
        self.arm_jids: dict[str, list[int]] = {}
        self.arm_cols: dict[str, torch.Tensor] = {}
        for side, regexes in (("left", cfg.left_arm_joint_names), ("right", cfg.right_arm_joint_names)):
            jids, _ = self.robot.find_joints(regexes)
            self.arm_jids[side] = jids
            self.arm_cols[side] = torch.tensor(
                [self.all_arm_joint_ids.index(j) for j in jids], device=self.device, dtype=torch.long
            )

        # End-effector bodies + jacobian indexing (floating base: body index
        # unshifted, joint columns offset by the 6 base DOFs).
        self.ee_body_idx: dict[str, int] = {}
        self.ee_jacobi_idx: dict[str, int] = {}
        self.jacobi_joint_cols: dict[str, list[int]] = {}
        for side, body_name in (("left", cfg.left_ee_body_name), ("right", cfg.right_ee_body_name)):
            idx = self.robot.find_bodies(body_name)[0][0]
            self.ee_body_idx[side] = idx
            if self.robot.is_fixed_base:
                self.ee_jacobi_idx[side] = idx - 1
                self.jacobi_joint_cols[side] = list(self.arm_jids[side])
            else:
                self.ee_jacobi_idx[side] = idx
                self.jacobi_joint_cols[side] = [j + 6 for j in self.arm_jids[side]]

        self.torso_body_idx = self.robot.find_bodies(cfg.torso_body_name)[0][0]

        # dp4c lean program (2026-08-07): the WISH — the world-frame point each
        # arm is TRYING to reach, stored pre-resolution. Consumed by the
        # arm_wish_b obs (the policy sees the wish even when the arm cannot
        # reach it) and desk_reach_bonus (leaning pays). desk_wish_mask marks
        # desk-plane draws (the reach reward is gated to them).
        self.wish_w = {
            "left": torch.zeros(env.num_envs, 3, device=env.device),
            "right": torch.zeros(env.num_envs, 3, device=env.device),
        }
        self.desk_wish_mask = {
            "left": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "right": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }

        # dp5 DEPLOY-MIMICKING ARM CYCLE (operator 2026-08-10): the real stack
        # does not wave the arms at random — ActionModule returns to its
        # go_to_start pose, reaches a desk point, reaches another, returns.
        # cycle_pattern encodes that as a repeating slot sequence per env.
        self._cycle_i = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        _codes = {"start": 0, "desk": 1, "free": 2}
        self._cycle_codes = torch.tensor(
            [_codes[k] for k in cfg.cycle_pattern], dtype=torch.long, device=env.device
        )

        # One batched DLS controller per arm. Position-only by default (the
        # 7-DOF arm resolves the extra DOFs minimally from the seed, like a
        # seeded KDL solve). orientation_mode=True (p12+) switches to 6-DoF
        # "pose" targets — deployment MoveIt is orientation-constrained, and
        # position-only training left the wrist/roll command dims unvisited
        # (the wrist_roll -2.4 OOD incident).
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose" if cfg.orientation_mode else "position",
            use_relative_mode=False, ik_method="dls",
        )
        self.ik = {
            "left": DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device),
            "right": DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device),
        }

        # Default arm pose (num_envs, 14) — also the peace-time target.
        dpos = self.robot.data.default_joint_pos
        self.default_arm_pos = dpos[:, self.all_arm_joint_ids].clone()

        # Soft joint limits with margin, over the 14 arm joints.
        lims = self.robot.data.soft_joint_pos_limits[:, self.all_arm_joint_ids, :]
        self.soft_lo = lims[..., 0] + cfg.soft_limit_margin
        self.soft_hi = lims[..., 1] - cfg.soft_limit_margin

        # The command: absolute arm joint targets. Starts at default.
        self.joint_targets = self.default_arm_pos.clone()

        # Per-arm Cartesian targets in the torso frame + per-env mode flags.
        self.target_pos_b = {
            "left": torch.zeros(env.num_envs, 3, device=env.device),
            "right": torch.zeros(env.num_envs, 3, device=env.device),
        }
        # Orientation targets (pose mode): quat per arm + the sampled local
        # delta, applied lazily to the MEASURED ee orientation on the first
        # update after a resample (body poses are stale at reset-time
        # resampling — same constraint as the position anchor).
        ident = torch.zeros(env.num_envs, 4, device=env.device)
        ident[:, 0] = 1.0
        self.target_quat_b = {"left": ident.clone(), "right": ident.clone()}
        self.orient_delta = {"left": ident.clone(), "right": ident.clone()}
        self.orient_pending = {
            "left": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "right": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }

        self.default_mode = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.frozen = {
            "left": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "right": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }
        self.time_since_resample = torch.zeros(env.num_envs, device=env.device)

        # Hand position at the DEFAULT pose in the torso frame — the anchor
        # the workspace offsets are sampled around. Captured lazily on the
        # first update (all envs sit at the default pose right after startup;
        # body poses are stale during reset-time resampling, so it cannot be
        # captured in _resample_command).
        self.default_ee_pos_b: dict[str, torch.Tensor] | None = None
        # Envs that resampled before the capture existed — re-rolled on capture.
        self._pending_resample = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        self.dt = env.step_dt

        self.metrics["pos_err_left"] = torch.zeros(env.num_envs, device=env.device)
        self.metrics["pos_err_right"] = torch.zeros(env.num_envs, device=env.device)

    def __str__(self) -> str:
        return (
            f"IKArmPoseCommand(dls position IK, resample={self.cfg.resampling_time_range}s, "
            f"default_pose_prob={self.cfg.default_pose_prob:.2f}, "
            f"workspace_scale={self.cfg.workspace_scale:.2f}, "
            f"max_joint_speed={self.cfg.max_joint_speed:.2f}rad/s)"
        )

    @property
    def command(self) -> torch.Tensor:
        """(num_envs, 14) absolute arm joint targets — same as UniformArmPoseCommand."""
        return self.joint_targets

    # ------------------------------------------------------------------
    # resampling
    # ------------------------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        if n == 0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        self.time_since_resample[env_ids] = 0.0
        self.frozen["left"][env_ids] = False
        self.frozen["right"][env_ids] = False

        # Peace-time draw: whole robot goes to (stays at) the default pose.
        self.default_mode[env_ids] = torch.rand(n, device=self.device) < self.cfg.default_pose_prob

        if self.default_ee_pos_b is None:
            # No workspace anchor yet (startup reset) — hold default until the
            # first update captures it, then re-roll these envs.
            self._pending_resample[env_ids] = True
            self.default_mode[env_ids] = True
            return

        self._sample_targets(env_ids)

    def _sample_targets(self, env_ids: torch.Tensor):
        """Draw per-arm Cartesian targets (torso frame) for env_ids.

        Three KINDS of draw:
          start : the deploy go_to_start pose (ABSOLUTE torso-frame point, the
                  colleague's raised hands-above-desk pose) — identity
                  orientation delta so the return is a clean repeatable posture
          desk  : a point on the desk plane near the anchor (work zone)
          free  : the old uniform draw inside workspace_offset (robustness)

        cycle_mode=False keeps the historical behaviour exactly: default_pose /
        desk_level_prob / free, decided per resample. cycle_mode=True walks
        cycle_pattern per env (e.g. start, desk, desk -> repeat), with
        cycle_random_prob of any slot being replaced by a free draw so the
        policy still sees off-script arm poses.
        """
        n = len(env_ids)
        s = float(self.cfg.workspace_scale)
        env = self._env

        # --- decide the KIND per env ---
        if self.cfg.cycle_mode:
            slot = self._cycle_i[env_ids] % len(self._cycle_codes)
            self._cycle_i[env_ids] += 1
            kind = self._cycle_codes[slot].clone()
            # a fraction of scheduled slots become free draws (robustness)
            if self.cfg.cycle_random_prob > 0.0:
                roll = torch.rand(n, device=self.device) < self.cfg.cycle_random_prob
                kind[roll] = 2
        else:
            kind = torch.full((n,), 2, dtype=torch.long, device=self.device)  # free
            desk_p = 0.0
            if (
                self.cfg.desk_level_prob > 0.0
                and hasattr(env, "spawn_root_xy")
                and hasattr(env, "spawn_yaw")
            ):
                desk_p = min(
                    1.0,
                    float(self.cfg.desk_level_prob)
                    / max(1e-6, 1.0 - float(self.cfg.default_pose_prob)),
                )
            if desk_p > 0.0:
                kind[torch.rand(n, device=self.device) < desk_p] = 1
        # desk slots need the spawn buffers; fall back to free if absent
        if not (hasattr(env, "spawn_root_xy") and hasattr(env, "spawn_yaw")):
            kind[kind == 1] = 2

        for side in ("left", "right"):
            sign = 1.0 if side == "left" else -1.0

            # ---- FREE draws (uniform inside the offset box) ----
            m_free = kind == 2
            if m_free.any():
                ids = env_ids[m_free]
                k = len(ids)
                lo = torch.tensor(self.cfg.workspace_offset[0], device=self.device) * s
                hi = torch.tensor(self.cfg.workspace_offset[1], device=self.device) * s
                offs = lo + torch.rand(k, 3, device=self.device) * (hi - lo)
                if side == "right":
                    offs[:, 1] = -offs[:, 1]
                self.target_pos_b[side][ids] = self.default_ee_pos_b[side][ids] + offs
                tp = self.robot.data.body_pose_w[ids, self.torso_body_idx]
                self.wish_w[side][ids] = tp[:, 0:3] + quat_apply(
                    tp[:, 3:7], self.target_pos_b[side][ids]
                )
                self.desk_wish_mask[side][ids] = False

            # ---- START pose (absolute torso-frame point, deploy analog) ----
            m_start = kind == 0
            if m_start.any():
                ids = env_ids[m_start]
                k = len(ids)
                sp = torch.tensor(self.cfg.start_pose_b, device=self.device).repeat(k, 1)
                sp[:, 1] = sp[:, 1] * sign          # y is OUTWARD, mirrored per side
                self.target_pos_b[side][ids] = sp
                tp = self.robot.data.body_pose_w[ids, self.torso_body_idx]
                self.wish_w[side][ids] = tp[:, 0:3] + quat_apply(tp[:, 3:7], sp)
                self.desk_wish_mask[side][ids] = False

            # ---- DESK plane near the anchor ----
            m_desk = kind == 1
            if m_desk.any():
                ids = env_ids[m_desk]
                k = len(ids)
                syaw = env.spawn_yaw[ids]
                fwd = torch.stack([torch.cos(syaw), torch.sin(syaw)], dim=-1)
                lat = torch.stack([-torch.sin(syaw), torch.cos(syaw)], dim=-1)
                anchor_xy = env.spawn_root_xy[ids] + self.cfg.desk_fwd_offset * fwd
                x_lo, x_hi = self.cfg.desk_x_range
                y_lo, y_hi = self.cfg.desk_y_range
                dx = x_lo + torch.rand(k, device=self.device) * (x_hi - x_lo)
                dy = y_lo + torch.rand(k, device=self.device) * (y_hi - y_lo)
                p_xy = anchor_xy + dx.unsqueeze(-1) * fwd + dy.unsqueeze(-1) * lat
                z = self.cfg.desk_height_w + (
                    torch.rand(k, device=self.device) * 2.0 - 1.0
                ) * self.cfg.desk_z_jitter
                p_w = torch.cat([p_xy, z.unsqueeze(-1)], dim=-1)
                tp = self.robot.data.body_pose_w[ids, self.torso_body_idx]
                pos_b, _ = subtract_frame_transforms(tp[:, 0:3], tp[:, 3:7], p_w)
                self.target_pos_b[side][ids] = pos_b
                self.wish_w[side][ids] = p_w
                self.desk_wish_mask[side][ids] = True

            # ---- dp5b slab clip: nothing may be commanded inside the table ----
            # An unreachable target under a kinematic slab is not a hard task,
            # it is an illegal order: the IK reports a residual the arm can
            # only answer by pressing, so desk_hit is paid every step for the
            # whole hold. Applied to ALL kinds (free/start/desk) after the fact
            # so one rule covers every draw path.
            if (
                self.cfg.desk_slab_clip
                and hasattr(env, "spawn_root_xy")
                and hasattr(env, "spawn_yaw")
            ):
                p_w = self.wish_w[side][env_ids]
                syaw = env.spawn_yaw[env_ids]
                fwd = torch.stack([torch.cos(syaw), torch.sin(syaw)], dim=-1)
                lat = torch.stack([-torch.sin(syaw), torch.cos(syaw)], dim=-1)
                anchor_xy = env.spawn_root_xy[env_ids] + self.cfg.desk_fwd_offset * fwd
                d = p_w[:, :2] - anchor_xy
                sx, sy = self.cfg.desk_slab_size
                over = ((d * fwd).sum(-1).abs() <= 0.5 * sx) & (
                    (d * lat).sum(-1).abs() <= 0.5 * sy
                )
                z_min = self.cfg.desk_slab_top + self.cfg.desk_clear_z
                bad = over & (p_w[:, 2] < z_min)
                if bad.any():
                    p_w[bad, 2] = z_min
                    self.wish_w[side][env_ids] = p_w
                    tp = self.robot.data.body_pose_w[env_ids, self.torso_body_idx]
                    pos_b, _ = subtract_frame_transforms(tp[:, 0:3], tp[:, 3:7], p_w)
                    self.target_pos_b[side][env_ids] = pos_b

            if self.cfg.orientation_mode:
                lim = torch.tensor(self.cfg.orientation_delta_rpy, device=self.device)
                rpy = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * lim
                constrained = torch.rand(n, device=self.device) < self.cfg.orientation_prob
                # START slots hold identity: a repeatable return posture, like
                # the deploy go_to_start which commands a fixed orientation.
                constrained = constrained & (kind != 0)
                rpy = rpy * constrained.unsqueeze(1).float()
                self.orient_delta[side][env_ids] = quat_from_euler_xyz(
                    rpy[:, 0], rpy[:, 1], rpy[:, 2]
                )
                self.orient_pending[side][env_ids] = True

    # ------------------------------------------------------------------
    # per-step update
    # ------------------------------------------------------------------

    def _ee_state_b(self, side: str):
        """EE position/orientation in the torso frame + torso world pose."""
        torso_pose_w = self.robot.data.body_pose_w[:, self.torso_body_idx]
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_idx[side]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            torso_pose_w[:, 0:3], torso_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        return ee_pos_b, ee_quat_b, torso_pose_w

    def _update_command(self):
        self.time_since_resample += self.dt

        # Lazy workspace-anchor capture (first update: everything at default).
        if self.default_ee_pos_b is None:
            self.default_ee_pos_b = {}
            for side in ("left", "right"):
                ee_pos_b, _, _ = self._ee_state_b(side)
                self.default_ee_pos_b[side] = ee_pos_b.clone()
            pending = torch.nonzero(self._pending_resample).flatten()
            if len(pending) > 0:
                # Re-roll the startup resamples that had no anchor yet.
                self.default_mode[pending] = (
                    torch.rand(len(pending), device=self.device) < self.cfg.default_pose_prob
                )
                ik_envs = pending[~self.default_mode[pending]]
                self._sample_targets(ik_envs)
                self._pending_resample[:] = False

        step_limit = self.cfg.max_joint_speed * self.dt

        for side in ("left", "right"):
            cols = self.arm_cols[side]
            jids = self.arm_jids[side]

            ee_pos_b, ee_quat_b, _ = self._ee_state_b(side)
            pos_err = torch.norm(self.target_pos_b[side] - ee_pos_b, dim=-1)
            # Peace-time envs track no Cartesian target — mask them out so the
            # logged error reflects only actively-IK'd hands.
            self.metrics[f"pos_err_{side}"] = torch.where(
                self.default_mode, torch.zeros_like(pos_err), pos_err
            )

            # Freeze a hand once it converged or ran out of settle time
            # (unreachable target -> hold the closest reached pose, like the
            # deployment's partial-reach bisection).
            self.frozen[side] |= pos_err < self.cfg.converge_tol
            self.frozen[side] |= self.time_since_resample > self.cfg.settle_time

            # Jacobian of this hand wrt its 7 arm joints, rotated into the
            # torso frame (targets and errors live there).
            jacobian = self.robot.root_physx_view.get_jacobians()[
                :, self.ee_jacobi_idx[side], :, :
            ][:, :, self.jacobi_joint_cols[side]]
            torso_quat_w = self.robot.data.body_pose_w[:, self.torso_body_idx, 3:7]
            rot = matrix_from_quat(quat_inv(torso_quat_w))
            jacobian = torch.cat(
                (torch.bmm(rot, jacobian[:, :3, :]), torch.bmm(rot, jacobian[:, 3:, :])), dim=1
            )

            joint_pos = self.robot.data.joint_pos[:, jids]
            ik = self.ik[side]
            if self.cfg.orientation_mode:
                # Lazily lock in the orientation target: measured ee quat at
                # the first update after resample, rotated by the sampled
                # local delta ("solve from where you are" — keeps 6-DoF
                # targets near-feasible while sweeping the wrist/roll dims).
                pend = self.orient_pending[side]
                if pend.any():
                    self.target_quat_b[side][pend] = quat_mul(
                        ee_quat_b[pend], self.orient_delta[side][pend]
                    )
                    self.orient_pending[side][:] = False
                ik.set_command(
                    torch.cat((self.target_pos_b[side], self.target_quat_b[side]), dim=1)
                )
            else:
                ik.set_command(self.target_pos_b[side], ee_quat=ee_quat_b)
            # The controller returns joint_pos + dq with dq = J^+ (x* - x).
            # Apply dq ON TOP OF the current TARGET (integral action), not on
            # the measured position: re-anchoring to joint_pos reaches
            # equilibrium where the IK correction merely cancels the PD
            # gravity sag, leaving a persistent Cartesian error (~sag, 0.2m+
            # at arm kp 40). Integrating on the target keeps pushing until
            # the PHYSICAL hand reaches the target, sag compensated.
            dq = ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos) - joint_pos

            cur_q = self.joint_targets[:, cols]
            # Peace-time envs head to the default pose instead of the IK goal.
            dq = torch.where(
                self.default_mode.unsqueeze(1), self.default_arm_pos[:, cols] - cur_q, dq
            )

            # Rate-limited integration; frozen hands hold.
            delta = torch.clamp(dq, -step_limit, step_limit)
            active = (~self.frozen[side] | self.default_mode).unsqueeze(1)
            new_q = cur_q + delta * active.float()
            new_q = torch.clamp(new_q, self.soft_lo[:, cols], self.soft_hi[:, cols])
            self.joint_targets[:, cols] = new_q

        if self.cfg.apply_directly:
            self.robot.set_joint_position_target(self.joint_targets, joint_ids=self.all_arm_joint_ids)

    def _update_metrics(self):
        pass  # per-side position errors are refreshed inside _update_command

    # ------------------------------------------------------------------
    # debug visualization — one dot per hand target
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                import isaaclab.sim as sim_utils
                from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

                self.target_visualizer = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path="/Visuals/Command/arm_ik_targets",
                        markers={
                            "left": sim_utils.SphereCfg(
                                radius=0.035,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
                            ),
                            "right": sim_utils.SphereCfg(
                                radius=0.035,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 1.0)),
                            ),
                        },
                    )
                )
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self.default_ee_pos_b is None:
            return
        torso_pose_w = self.robot.data.body_pose_w[:, self.torso_body_idx]
        rot = matrix_from_quat(torso_pose_w[:, 3:7])
        pos_w = []
        for side in ("left", "right"):
            # For peace-time envs the "target" is wherever the default hand sits.
            tgt_b = torch.where(
                self.default_mode.unsqueeze(1), self.default_ee_pos_b[side], self.target_pos_b[side]
            )
            pos_w.append(torso_pose_w[:, 0:3] + torch.bmm(rot, tgt_b.unsqueeze(-1)).squeeze(-1))
        translations = torch.cat(pos_w, dim=0)
        # derive the env count from a tensor in hand — self._env.num_envs touches
        # env.scene, which is already destroyed when the render callback fires one
        # last time during env.close() (teardown AttributeError, 2026-07-26)
        n = self.default_mode.shape[0]
        indices = torch.cat(
            (
                torch.zeros(n, dtype=torch.long, device=self.device),
                torch.ones(n, dtype=torch.long, device=self.device),
            )
        )
        self.target_visualizer.visualize(translations=translations, marker_indices=indices)


@configclass
class IKArmPoseCommandCfg(CommandTermCfg):
    """Cfg for IKArmPoseCommand (the IK envelope)."""

    class_type: type = IKArmPoseCommand

    asset_name: str = MISSING
    all_arm_joint_names: list[str] = MISSING
    """All 14 arm joints — MUST be the same regex list as the old command so
    the observation layout matches (warmstart compatibility)."""

    left_arm_joint_names: list[str] = ["left_shoulder_.*", "left_elbow.*", "left_wrist.*"]
    right_arm_joint_names: list[str] = ["right_shoulder_.*", "right_elbow.*", "right_wrist.*"]

    left_ee_body_name: str = "left_wrist_yaw_link"
    right_ee_body_name: str = "right_wrist_yaw_link"
    torso_body_name: str = "torso_link"
    """IK targets are expressed in this body's frame — matches the deployment
    ActionModule, which solves MoveIt IK in torso_link."""

    workspace_offset: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.05, -0.15, -0.10),
        (0.55, 0.35, 0.85),
    )
    """(lo, hi) Cartesian offset box (m) added to the default hand position in
    the torso frame, for the LEFT arm: x forward, y outward (mirrored for the
    right arm; negative y = cross-body), z up. Defaults span thigh-level to
    above-shoulder reach. Unreachable corners are fine — the hand stretches
    toward them and freezes at settle_time (deployment partial-reach analog)."""

    workspace_scale: float = 1.0
    """Multiplier on the offset box — the curriculum knob (ramp small→full)."""

    default_pose_prob: float = 0.25
    """Peace-time anchor: probability a resample commands the default pose."""

    max_joint_speed: float = 0.6
    """Rate limit (rad/s) on the joint targets — the transition speed between
    held poses, standing in for the deployment-side interpolation (~4s ramps).
    1.5 rad/s proved violent enough to knock over a wobble-trained policy."""

    converge_tol: float = 0.02
    """Hand-to-target distance (m) below which the hand's command freezes."""

    settle_time: float = 6.0
    """Seconds after a resample before the command freezes regardless —
    bounds the stretch toward unreachable targets."""

    soft_limit_margin: float = 0.02
    """Margin (rad) kept inside each soft joint limit."""

    # --- 6-DoF orientation targets (p12+; deployment MoveIt is orientation-
    # constrained, position-only training left wrist/roll command dims
    # unvisited — the wrist_roll -2.4 OOD incident) ---
    orientation_mode: bool = False
    """Solve 6-DoF pose targets instead of position-only."""

    orientation_delta_rpy: tuple[float, float, float] = (2.4, 0.8, 0.8)
    """Max |roll,pitch,yaw| (rad) of the local rotation delta applied to the
    measured ee orientation at resample. Roll span covers the observed
    wrist_roll ±2.4 OOD spec."""

    orientation_prob: float = 0.7
    """Fraction of resamples that constrain orientation (identity delta
    otherwise — those behave like position-only draws)."""

    # --- desk-level draws (dp4c: balance while the arms WORK AT THE DESK) ---
    desk_level_prob: float = 0.0
    """UNCONDITIONAL fraction of resamples whose target lies on the desk plane
    near the anchor (operator 2026-08-06 split: default/desk/random =
    20/30/50 -> default_pose_prob 0.20, desk_level_prob 0.30). 0 disables —
    existing tasks unchanged. Needs capture_spawn_state buffers (silently
    falls back to the free draw until they exist)."""

    desk_fwd_offset: float = 0.5
    """Anchor forward offset (m) from spawn along the spawn heading — MUST
    match anchor_point_b / anchor_hold_bonus (0.5 in the desk contract)."""

    desk_height_w: float = 1.0
    """World z (m) of the desk plane — MUST match anchor_point_b's height_w."""

    desk_x_range: tuple[float, float] = (-0.20, 0.10)
    """Offset (m) along spawn-forward relative to the ANCHOR (desk centre):
    -0.20 = near edge, +0.10 = past centre."""

    desk_y_range: tuple[float, float] = (-0.30, 0.30)
    """Lateral offset (m) relative to the anchor (both arms, same range —
    cross-body corners freeze at settle_time like any unreachable draw)."""

    desk_z_jitter: float = 0.05
    """Uniform +- jitter (m) on the desk-plane z (objects sit ON the desk).
    dp5b: keep `desk_height_w - desk_z_jitter` STRICTLY ABOVE `desk_slab_top`.
    dp5 shipped 1.00 +- 0.05 against a slab topping out at 1.00, so half of
    every desk draw was a point INSIDE the table (visible in the previews as
    arms pressing up from underneath) and the wrists paid desk_hit forever."""

    # --- dp5b: no target may be inside or under the slab ---
    desk_slab_clip: bool = False
    """Raise any target whose xy falls over the desk footprint to at least
    `desk_slab_top + desk_clear_z`. Covers FREE and START draws too, which is
    where dp4c_deskcol2's -0.587 desk_hit came from at desk_level_prob=0."""

    desk_slab_top: float = 1.0
    """World z (m) of the physical slab's TOP face. Distinct from
    `desk_height_w` (the draw plane), which dp5b lifts above it."""

    desk_slab_size: tuple[float, float] = (0.5, 2.0)
    """Slab footprint (x, y) in the SPAWN frame — must match scene.desk."""

    desk_clear_z: float = 0.04
    """Clearance (m) held above the slab top when clipping."""

    # --- dp5 deploy-mimicking cycle ---
    cycle_mode: bool = False
    """Walk `cycle_pattern` per env instead of drawing a kind at random."""

    cycle_pattern: tuple = ("start", "desk", "desk")
    """Repeating slot sequence. Mirrors the real ActionModule loop: return to
    go_to_start, reach a desk point, reach another, return."""

    cycle_random_prob: float = 0.2
    """Probability a scheduled slot is replaced by a FREE draw — keeps
    off-script arm poses in the distribution for robustness."""

    start_pose_b: tuple = (0.250, 0.490, 0.550)
    """The deploy go_to_start pose as an ABSOLUTE torso-frame point (y is
    OUTWARD, mirrored for the right arm). From ActionModule
    high_level_sdk.go_to_start: left move(0.250, 0.490, 0.550, roll 103.5,
    pitch -41.5); the same xyz for the right arm with its own orientation."""

    apply_directly: bool = True
    resampling_time_range: tuple[float, float] = (2.0, 10.0)
    debug_vis: bool = False
