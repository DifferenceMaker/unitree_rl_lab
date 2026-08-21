"""reward_ledger — LIVE per-term reward recomputation for MuJoCo sim2sim.

Operator design 2026-08-21: the Isaac reward HUD's ledger, recomputed in the
rig. In Isaac the ledger is a by-product of env.step(); the MuJoCo stack has
no RewardManager, so this module re-implements the desk line's POSE-BASED
terms (the 2-3deg-lean suspects) from:

    rt/lowstate       IMU quat/gyro (already consumed by balance_metrics)
    rt/sim_base_pose  ground-truth base pose/vel + wrist positions
                      (simulate main.cc publisher, 2026-08-21 — sim2sim-only
                      privilege; JSON {"p","q","v","w","lw","rw"})
    rt/anchor_point   anchor in BASE frame ({"p":[x,y,z]})
    rt/lowcmd         controller joint targets (action_rate)

Weights/params are parsed from the ACTIVE policy's params/env.yaml (regex on
the yaml text — no torch/isaac imports; tv-env friendly), so the ledger shows
the run's own economy. Known approximations (marked ~ in the HUD):
  * anchor_hold: pos error taken as |anchor_rel_b_xy - (fwd_offset,0)| (exact
    when heading ~ spawn yaw); yaw term omitted (spawn yaw unavailable).
  * torso_stability_bonus: product form exp(-|v_xy|^2/sl^2)*exp(-|w|^2/sa^2).
  * torso_ang_vel standing gate assumed ON (desk velocity command ~ 0).
  * NOT implemented (v2): joint deviations (SDK<->Isaac index map), desk_reach
    (needs the arm-target world points), contact terms (undesired, capture,
    slide, gait) — those need a sim contact export.

Every tick is appended to a tape; on exit the tape is written as JSON
(names, weights, rows, dt) next to the sidecar log — rehud-style post-analysis
without any replay machinery.
"""
import json
import math
import os
import re
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_


def _yaml_num(text, pattern, default=None):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default


class RewardLedger:
    def __init__(self, env_yaml_path: str, joint_names=None):
        txt = open(env_yaml_path).read()

        def term_weight(name):
            m = re.search(rf"^  {name}:\n(?:    .*\n)*?    weight: (-?[0-9.e+-]+)",
                          txt, re.M)
            return float(m.group(1)) if m else None

        self.w = {n: term_weight(n) for n in (
            "alive", "anchor_hold", "upright_bonus", "flat_orientation_l2",
            "base_height", "torso_stability_bonus", "torso_lin_vel_xy",
            "torso_ang_vel", "track_lin_vel_xy", "track_ang_vel_z", "action_rate",
            "joint_deviation_hips", "joint_deviation_torso", "joint_deviation_knees",
            "desk_reach", "undesired_contacts", "desk_hit",
            "feet_slide", "feet_too_near",
            "base_linear_velocity", "base_angular_velocity",
        )}
        # drop zero-weight terms (e.g. desk_hit 0 in the dp5 line)
        for k, v in list(self.w.items()):
            if v == 0.0:
                self.w[k] = None

        # ---- joint deviations (v3): SDK-order joint names from the sidecar's
        # FK model + Isaac-order defaults mapped via deploy.yaml joint_ids_map
        self._dev_groups = {}
        self._q_default_sdk = None
        dep_path = os.path.join(os.path.dirname(env_yaml_path), "deploy.yaml")
        if joint_names and os.path.isfile(dep_path):
            dep = open(dep_path).read()

            def _num_list(key):
                m = re.search(rf"^{key}: \[([^\]]+)\]", dep, re.M | re.S)
                return [float(x) for x in m.group(1).replace("\n", " ").split(",")] if m else None

            ids_map = _num_list("joint_ids_map")
            q_def_isaac = _num_list("default_joint_pos")
            if ids_map and q_def_isaac and len(joint_names) == len(q_def_isaac):
                self._q_default_sdk = np.zeros(len(joint_names))
                for i, sdk in enumerate(ids_map):
                    self._q_default_sdk[int(sdk)] = q_def_isaac[i]
                for gname, pat in (
                        ("joint_deviation_hips", r"hip_(roll|yaw)"),
                        ("joint_deviation_torso", r"torso"),
                        ("joint_deviation_knees", r"knee")):
                    idx = [j for j, n in enumerate(joint_names) if re.search(pat, n)]
                    if idx:
                        self._dev_groups[gname] = np.array(idx, dtype=int)
        self.p = {
            "anchor_sigma": _yaml_num(txt, r"anchor_hold:\n(?:    .*\n)*?      sigma: ([0-9.]+)", 0.15),
            "anchor_fwd": _yaml_num(txt, r"anchor_hold:\n(?:    .*\n)*?      fwd_offset: ([0-9.]+)", 0.5),
            "upright_std": _yaml_num(txt, r"upright_bonus:\n(?:    .*\n)*?      std: ([0-9.]+)", 0.05),
            "height_target": _yaml_num(txt, r"base_height:\n(?:    .*\n)*?      target_height: ([0-9.]+)", 0.87),
            "reach_sigma": _yaml_num(txt, r"desk_reach:\n(?:    .*\n)*?      sigma: ([0-9.]+)", 0.25),
            "near_thresh": _yaml_num(txt, r"feet_too_near:\n(?:    .*\n)*?      threshold: ([0-9.]+)", 0.18),
            "track_std2": 0.25,   # std = sqrt(0.25) in the trunk
            "stab_sl": 0.15, "stab_sa": 0.30,
        }
        self.names = [n for n, w in self.w.items() if w is not None]

        self._pose = None          # dict from rt/sim_base_pose
        self._anchor = None        # [x,y,z] base frame
        self._prev_cmd = None
        self._dq_cmd2 = 0.0
        self._prev_feet = None
        self._prev_feet_t = 0.0
        self.rows = []
        self.t0 = time.monotonic()
        self._t_last = self.t0

        self._pose_sub = ChannelSubscriber("rt/sim_base_pose", String_)
        self._pose_sub.Init(self._on_pose, 10)
        self._anchor_sub = ChannelSubscriber("rt/anchor_point", String_)
        self._anchor_sub.Init(self._on_anchor, 10)
        try:
            self._cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
            self._cmd_sub.Init(self._on_lowcmd, 10)
        except Exception:
            self._cmd_sub = None
        print(f"[LEDGER] {len(self.names)} terms from {env_yaml_path}: "
              f"{', '.join(self.names)}", flush=True)

    # ---- subscribers ----
    def _on_pose(self, msg):
        try:
            self._pose = json.loads(msg.data)
        except Exception:
            pass

    def _on_anchor(self, msg):
        try:
            self._anchor = json.loads(msg.data)["p"]
        except Exception:
            pass

    def _on_lowcmd(self, msg):
        try:
            q = np.array([mc.q for mc in msg.motor_cmd[:27]], dtype=np.float64)
            if self._prev_cmd is not None and q.shape == self._prev_cmd.shape:
                d = q - self._prev_cmd
                self._dq_cmd2 = float(np.dot(d, d))
            self._prev_cmd = q
        except Exception:
            pass

    # ---- math ----
    @staticmethod
    def _quat_rot(q, v):
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])
        return R @ np.asarray(v)

    def tick(self, proj_grav_xy2: float, q=None):
        """Compute all terms. proj_grav_xy2 = |projected_gravity_b xy|^2 from
        the sidecar's IMU path (already computed there). Returns [(name, v)]."""
        vals = {}
        w, p = self.w, self.p
        if w.get("alive") is not None:
            vals["alive"] = w["alive"]
        if w.get("upright_bonus") is not None:
            vals["upright_bonus"] = w["upright_bonus"] * math.exp(
                -proj_grav_xy2 / (p["upright_std"] ** 2))
        if w.get("flat_orientation_l2") is not None:
            vals["flat_orientation_l2"] = w["flat_orientation_l2"] * proj_grav_xy2
        if self._anchor is not None and w.get("anchor_hold") is not None:
            dx = self._anchor[0] - p["anchor_fwd"]
            dy = self._anchor[1]
            vals["anchor_hold~"] = w["anchor_hold"] * math.exp(
                -(dx*dx + dy*dy) / (p["anchor_sigma"] ** 2))
        if self._pose is not None:
            ps = self._pose
            z = ps["p"][2]
            v_xy2 = ps["v"][0]**2 + ps["v"][1]**2
            w_world = self._quat_rot(ps["q"], ps["w"])   # body ang -> world
            w2 = float(np.dot(w_world, w_world))
            if w.get("base_height") is not None:
                vals["base_height"] = w["base_height"] * (z - p["height_target"])**2
            if w.get("torso_lin_vel_xy") is not None:
                vals["torso_lin_vel_xy"] = w["torso_lin_vel_xy"] * v_xy2
            if w.get("torso_ang_vel") is not None:
                vals["torso_ang_vel~"] = w["torso_ang_vel"] * w2
            if w.get("torso_stability_bonus") is not None:
                vals["torso_stability_bonus~"] = w["torso_stability_bonus"] * math.exp(
                    -v_xy2 / p["stab_sl"]**2) * math.exp(-w2 / p["stab_sa"]**2)
            if w.get("track_lin_vel_xy") is not None:
                vals["track_lin_vel_xy"] = w["track_lin_vel_xy"] * math.exp(
                    -v_xy2 / p["track_std2"])
            if w.get("track_ang_vel_z") is not None:
                vals["track_ang_vel_z"] = w["track_ang_vel_z"] * math.exp(
                    -(w_world[2]**2) / p["track_std2"])
        if w.get("action_rate") is not None and self._prev_cmd is not None:
            vals["action_rate"] = w["action_rate"] * self._dq_cmd2

        # ---- v3 terms ----
        if q is not None and self._q_default_sdk is not None:
            dq = np.abs(np.asarray(q) - self._q_default_sdk)
            for gname, idx in self._dev_groups.items():
                if w.get(gname) is not None:
                    vals[gname] = w[gname] * float(dq[idx].sum())
        if self._pose is not None:
            ps = self._pose
            if w.get("base_linear_velocity") is not None:
                vals["base_linear_velocity"] = w["base_linear_velocity"] * ps["v"][2] ** 2
            if w.get("base_angular_velocity") is not None:
                ww = self._quat_rot(ps["q"], ps["w"])
                vals["base_angular_velocity"] = w["base_angular_velocity"] * float(
                    ww[0] ** 2 + ww[1] ** 2)
            # desk_reach: per-hand kernel on |wrist - target ball|, active when
            # the ball is visible (hidden balls park at z < 0)
            if w.get("desk_reach") is not None and "tl" in ps and "lw" in ps:
                tot = 0.0
                for wk, tk in (("lw", "tl"), ("rw", "tr")):
                    t = ps.get(tk)
                    if t and t[2] > 0.0:
                        e2 = sum((ps[wk][i] - t[i]) ** 2 for i in range(3))
                        tot += math.exp(-e2 / (self.p["reach_sigma"] ** 2))
                vals["desk_reach~"] = w["desk_reach"] * tot
            # undesired_contacts: COUNT of scoped bodies (pelvis/torso/hips/
            # knees) in >1N contact with the DESK (Isaac form = count)
            if w.get("undesired_contacts") is not None and "ucnt" in ps:
                vals["undesired_contact~"] = w["undesired_contacts"] * float(ps["ucnt"])
            # feet_slide: foot xy speed while in contact (positions differenced
            # at the publish rate)
            if w.get("feet_slide") is not None and "fl" in ps and "fc" in ps:
                slide = 0.0
                nowp = (ps["fl"], ps["fr"])
                if self._prev_feet is not None:
                    dt = max(1e-3, time.monotonic() - self._prev_feet_t)
                    for k in range(2):
                        vxy = math.hypot(nowp[k][0] - self._prev_feet[k][0],
                                         nowp[k][1] - self._prev_feet[k][1]) / dt
                        if ps["fc"][k] > 1.0:
                            slide += vxy
                self._prev_feet = (list(nowp[0]), list(nowp[1]))
                self._prev_feet_t = time.monotonic()
                vals["feet_slide~"] = w["feet_slide"] * slide
            # feet_too_near: hinge below the threshold on 3D foot distance
            if w.get("feet_too_near") is not None and "fl" in ps:
                dist = math.sqrt(sum((ps["fl"][i] - ps["fr"][i]) ** 2 for i in range(3)))
                vals["feet_too_near"] = w["feet_too_near"] * max(
                    0.0, self.p["near_thresh"] - dist)

        now = time.monotonic()
        self.rows.append([now - self.t0] + [vals.get(k, 0.0) for k in self._row_keys(vals)])
        self._t_last = now
        return sorted(vals.items(), key=lambda kv: -abs(kv[1]))

    _keys = None

    def _row_keys(self, vals):
        if self._keys is None:
            self._keys = sorted(vals.keys())
        return self._keys

    # ---- outputs ----
    def hud_field(self, items, top=12):
        """Structured '|'-separated 'name:value:frac' rows in a FIXED order
        (by |weight| desc, set once — rows never switch places; operator
        gauges-v2 feedback 2026-08-21). frac = value / |weight| clamped to
        [-1,1]: kernel incomes read as fill-fraction of their max; penalties
        can exceed and clamp."""
        if not hasattr(self, "_order"):
            self._order = sorted((n for n in self.names),
                                 key=lambda n: -abs(self.w[n] or 0.0))
        d = dict((n.rstrip("~"), (n, v)) for n, v in items)
        total = sum(v for _, v in items)
        rows = [f"TOTAL:{total:+.2f}:{max(-1.0, min(1.0, total / 50.0)):+.3f}"]
        for base in self._order[:top]:
            if base not in d:
                continue
            disp, v = d[base]
            scale = abs(self.w.get(base) or 1.0) or 1.0
            frac = max(-1.0, min(1.0, v / scale))
            rows.append(f"{disp[:18]}:{v:+.2f}:{frac:+.3f}")
        # wrist extension info row (desk_reach proxy) — frac 0 = text-ish row
        if self._pose is not None and "lw" in self._pose:
            b = self._pose["p"]
            lw = self._pose["lw"]; rw = self._pose["rw"]
            rows.append(f"wristfwd L{lw[0]-b[0]:+.2f} R{rw[0]-b[0]:+.2f}:0.00:0.000")
        return "|".join(rows)

    def save_tape(self, path):
        if not self.rows or self._keys is None:
            return None
        out = {"names": self._keys, "weights": {k: self.w.get(k.rstrip('~'))
               for k in self._keys}, "rows": self.rows,
               "note": "cols: [t_s] + names; values weighted /s; ~ = approximated term"}
        with open(path, "w") as f:
            json.dump(out, f)
        return path
