#!/usr/bin/env python3
"""Balance metrics sidecar for the H1-2 sim2sim harness (and the real robot).

A standalone process that subscribes to the same DDS lowstate the controller
reads (``rt/lowstate``, unitree_hg LowState_) and turns the eyeball harness
into a measurement tool. Deliberately a SIDECAR rather than in-controller
metrics: the production controller stays untouched, and because everything is
inferred from lowstate (never from MuJoCo ground truth), the identical script
runs against the real robot — same metrics on hardware, same print format.

Metrics (printed every --window samples, default 250 @ 50 Hz = 5 s):

  [METRICS] window=250 steps (5.0s) / touchdowns L=3 R=2 total=5 = 0.42/s \
/ feet_dist mean 0.328 min 0.310 / torso_ang_vel RMS 0.142

  * touchdowns  — cumulative per-foot touchdown count since start (or the
                  last `zero`). A touchdown is the rising edge of foot
                  contact, inferred by FK + hysteresis (lowstate has no foot
                  force for H1-2): foot height relative to the LOWER foot,
                  lifted when > LIFT_THRESH, grounded again when
                  < GROUND_THRESH. The rate is cumulative_total / elapsed
                  since start-or-zero (matches a whole-eval Isaac rate).
  * feet_dist   — 3D distance between the left/right ankle_roll body frames
                  (FK from measured joint angles; invariant to base pose).
                  mean and min over the window.
  * torso_ang_vel RMS — sqrt(mean ||imu gyro||^2) over the window.

Falls: tilt of the gravity axis > 1.0 rad (matches the controller's
bad_orientation check) on a rising edge → counted + printed as [FALL].

stdin commands while running:
  zero          zero all counters/accumulators mid-run (also SIGUSR1)
  mode <label>  annotate which disturbance mode is active (for the summary —
                the sidecar cannot see the controller's DPad state)
  note <text>   timestamped marker in the log
  quit          exit with summary (also Ctrl+C)

Usage (sim2sim):
  conda activate tv
  python3 balance_metrics.py                 # iface lo, auto-found XML
Usage (real robot, on pc4):
  python3 balance_metrics.py --iface eth0 --xml /path/to/h1_2.xml
"""

import argparse
import math
import os
import select
import signal
import sys
import threading
import time

import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("mujoco python package required (pip install mujoco)")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

NUM_JOINTS = 27  # H1-2 handless: 12 legs + torso + 14 arms (SDK order)

# Touchdown hysteresis on foot height relative to the lower foot (meters).
LIFT_THRESH = 0.03
GROUND_THRESH = 0.012

FALL_TILT_RAD = 1.0  # matches isaaclab::mdp::bad_orientation(env, 1.0)


def find_default_xml() -> str:
    # repos/unitree_rl_lab/deploy/robots/h1_2/tools/ -> repos/
    repos = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
    return os.path.join(repos, "unitree_mujoco", "unitree_robots", "h1_2", "h1_2.xml")


def quat_to_rotmat(q):
    """wxyz quaternion -> rotation matrix (base->world)."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-8:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class LowStateBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._msg = None
        self._stamp = 0.0

    def update(self, msg: LowState_):
        with self._lock:
            self._msg = msg
            self._stamp = time.monotonic()

    def latest(self):
        with self._lock:
            return self._msg, self._stamp


class FootState:
    """Hysteresis contact state machine for one foot."""

    def __init__(self):
        self.grounded = True  # robot starts standing
        self.touchdowns = 0

    def update(self, rel_height: float):
        if self.grounded:
            if rel_height > LIFT_THRESH:
                self.grounded = False
        else:
            if rel_height < GROUND_THRESH:
                self.grounded = True
                self.touchdowns += 1
                return True
        return False


class Metrics:
    def __init__(self, window: int, dt: float, mode_label: str):
        self.window = window
        self.dt = dt
        self.mode_label = mode_label
        self.publisher = None  # optional DDS publisher (rt/balance_metrics) for the sim HUD
        self.zero()

    def zero(self):
        self.t0 = time.monotonic()
        self.feet = {"L": FootState(), "R": FootState()}
        self.falls = 0
        self._fallen = False
        # window accumulators
        self.win_samples = 0
        self.win_t0 = self.t0
        self.win_dist = []
        self.win_gyro_sq = []
        # run accumulators (for the exit summary)
        self.run_dist_sum = 0.0
        self.run_dist_n = 0
        self.run_gyro_sq_sum = 0.0
        self.run_gyro_n = 0
        # projected_gravity (lean): gx=fwd/back, gy=lateral L/R, gz≈-1 upright
        self.win_pg = []
        self.run_pg_sum = np.zeros(3)
        self.run_pg_n = 0
        # raw IMU rpy (firmware Euler, rad): roll=lateral tilt, pitch=fwd tilt, yaw
        self.win_rpy = []
        self.run_rpy_sum = np.zeros(3)
        self.run_rpy_n = 0

    def step(self, dist, rel_h_l, rel_h_r, gyro, tilt, proj_grav, rpy=None):
        for key, rel_h in (("L", rel_h_l), ("R", rel_h_r)):
            self.feet[key].update(rel_h)

        gyro_sq = float(np.dot(gyro, gyro))
        pg = np.asarray(proj_grav, dtype=float)
        self.win_dist.append(dist)
        self.win_gyro_sq.append(gyro_sq)
        self.win_pg.append(pg)
        self.run_dist_sum += dist
        self.run_dist_n += 1
        self.run_gyro_sq_sum += gyro_sq
        self.run_gyro_n += 1
        self.run_pg_sum += pg
        self.run_pg_n += 1
        if rpy is not None:
            r = np.asarray(rpy, dtype=float)
            self.win_rpy.append(r)
            self.run_rpy_sum += r
            self.run_rpy_n += 1
        self.win_samples += 1

        if tilt > FALL_TILT_RAD and not self._fallen:
            self._fallen = True
            self.falls += 1
            print(f"[FALL] #{self.falls} tilt={tilt:.2f} rad "
                  f"t={time.monotonic() - self.t0:.1f}s", flush=True)
        elif tilt < FALL_TILT_RAD * 0.5:
            self._fallen = False

        if self.win_samples >= self.window:
            self._print_window()
            self.win_samples = 0
            self.win_t0 = time.monotonic()
            self.win_dist = []
            self.win_gyro_sq = []
            self.win_pg = []
            self.win_rpy = []

    def _print_window(self):
        now = time.monotonic()
        win_s = now - self.win_t0
        elapsed = now - self.t0
        td_l = self.feet["L"].touchdowns
        td_r = self.feet["R"].touchdowns
        total = td_l + td_r
        rate = total / elapsed if elapsed > 0 else 0.0
        mean_d = float(np.mean(self.win_dist)) if self.win_dist else float("nan")
        min_d = float(np.min(self.win_dist)) if self.win_dist else float("nan")
        rms = math.sqrt(float(np.mean(self.win_gyro_sq))) if self.win_gyro_sq else float("nan")
        if self.win_pg:
            pg = np.stack(self.win_pg); pgm = pg.mean(0); pgs = pg.std(0)
            lean_fwd = math.degrees(math.atan2(pgm[0], -pgm[2]))
            lean_lat = math.degrees(math.atan2(pgm[1], -pgm[2]))
        else:
            pgm = np.full(3, float("nan")); pgs = pgm; lean_fwd = lean_lat = float("nan")
        if self.win_rpy:
            rpy_m = np.stack(self.win_rpy).mean(0)
            imu_roll = math.degrees(rpy_m[0]); imu_pitch = math.degrees(rpy_m[1])
        else:
            imu_roll = imu_pitch = float("nan")
        print(f"[METRICS] window={self.window} steps ({win_s:.1f}s) / "
              f"touchdowns L={td_l} R={td_r} total={total} = {rate:.2f}/s / "
              f"feet_dist mean {mean_d:.3f} min {min_d:.3f} / "
              f"torso_ang_vel RMS {rms:.3f} / "
              f"proj_grav [{pgm[0]:+.3f},{pgm[1]:+.3f},{pgm[2]:+.3f}] "
              f"lean fwd={lean_fwd:+.1f} lat={lean_lat:+.1f}deg "
              f"(wander gx={pgs[0]:.3f} gy={pgs[1]:.3f}) "
              f"imu_rpy roll={imu_roll:+.1f} pitch={imu_pitch:+.1f}deg", flush=True)

        # Publish to the MuJoCo sim HUD (rt/balance_metrics). Best-effort; the sim
        # subscribes and overlays this. No effect on logging if no subscriber.
        if self.publisher is not None:
            payload = (
                '{"lean_fwd":%.1f,"lean_lat":%.1f,"steps_l":%d,"steps_r":%d,'
                '"touchdown_rate":%.2f,"torso_ang_vel_rms":%.3f}'
            ) % (lean_fwd, lean_lat, td_l, td_r, rate, rms)
            # cache the fields so the 5 Hz ledger publisher can carry them —
            # a ledger-only payload would blank the METRICS lines in the sim
            # overlay (set_metrics_from_json rebuilds from present fields).
            self.last_payload_fields = payload[1:-1]
            try:
                self.publisher.Write(String_(data=payload))
            except Exception:
                pass

    def summary(self):
        elapsed = time.monotonic() - self.t0
        td_l = self.feet["L"].touchdowns
        td_r = self.feet["R"].touchdowns
        total = td_l + td_r
        rate = total / elapsed if elapsed > 0 else 0.0
        mean_d = self.run_dist_sum / self.run_dist_n if self.run_dist_n else float("nan")
        rms = math.sqrt(self.run_gyro_sq_sum / self.run_gyro_n) if self.run_gyro_n else float("nan")
        pgm = (self.run_pg_sum / self.run_pg_n) if self.run_pg_n else np.full(3, float("nan"))
        lean_fwd = math.degrees(math.atan2(pgm[0], -pgm[2])) if self.run_pg_n else float("nan")
        lean_lat = math.degrees(math.atan2(pgm[1], -pgm[2])) if self.run_pg_n else float("nan")
        print("\n================ RUN SUMMARY ================", flush=True)
        print(f"  duration        : {elapsed:.1f} s "
              f"({self.run_dist_n} samples @ {1.0 / self.dt:.0f} Hz nominal)")
        print(f"  disturbance mode: {self.mode_label}")
        print(f"  touchdowns      : L={td_l} R={td_r} total={total} ({rate:.2f}/s)")
        print(f"  falls           : {self.falls}")
        print(f"  feet_dist mean  : {mean_d:.3f} m")
        print(f"  torso_ang_vel   : RMS {rms:.3f} rad/s")
        print(f"  proj_grav (mean): [{pgm[0]:+.3f}, {pgm[1]:+.3f}, {pgm[2]:+.3f}]")
        print(f"  lean (mean)     : fwd {lean_fwd:+.1f}deg  lateral {lean_lat:+.1f}deg")
        if self.run_rpy_n:
            rpy_m = self.run_rpy_sum / self.run_rpy_n
            print(f"  IMU rpy (mean)  : roll {math.degrees(rpy_m[0]):+.2f}deg  "
                  f"pitch {math.degrees(rpy_m[1]):+.2f}deg  yaw {math.degrees(rpy_m[2]):+.2f}deg")
            print("  >> IMU-bias test: compare IMU roll (and lean lateral) above to the iPhone")
            print("     PHYSICAL torso angle while standing — body more tilted than the IMU reads")
            print("     => IMU under-reports that tilt; the gap = ROLL_BIAS to apply.")
        print("=============================================", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="lo", help="DDS network interface (lo=sim, eth0=pc4)")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain id")
    ap.add_argument("--xml", default=None, help="h1_2 MJCF for FK (default: auto from repo layout)")
    ap.add_argument("--hz", type=float, default=50.0,
                    help="metric sample rate; 50 Hz = one sample per control step")
    ap.add_argument("--window", type=int, default=250, help="samples per [METRICS] print")
    ap.add_argument("--mode", default="(unset — use stdin: mode <label>)",
                    help="disturbance-mode label for the run summary")
    ap.add_argument("--ledger", default=None, metavar="ENV_YAML",
                    help="LIVE reward ledger: path to the policy's params/env.yaml "
                         "(weights parsed from it). Requires the sim's "
                         "rt/sim_base_pose publisher (2026-08-21+ build). Rows are "
                         "overlaid in the sim HUD and a tape JSON is saved on exit.")
    args = ap.parse_args()

    xml = args.xml or find_default_xml()
    if not os.path.isfile(xml):
        sys.exit(f"MJCF not found: {xml} (pass --xml)")

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)

    # Hinge joints in MJCF document order == SDK motor order (the h1_2 XML is
    # authored leg-major in SDK order; verified against BODY_JOINT_ORDER).
    hinge_joints = [j for j in range(model.njnt)
                    if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    if len(hinge_joints) != NUM_JOINTS:
        sys.exit(f"expected {NUM_JOINTS} non-free joints in {xml}, found {len(hinge_joints)}")
    qpos_addr = [model.jnt_qposadr[j] for j in hinge_joints]

    free_joint = next((j for j in range(model.njnt)
                       if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE), None)
    base_qpos = model.jnt_qposadr[free_joint] if free_joint is not None else None

    bid_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    bid_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    if bid_l < 0 or bid_r < 0:
        sys.exit("ankle_roll bodies not found in model")

    print(f"[SIDECAR] model: {xml}")
    print("[SIDECAR] SDK→joint mapping (verify against BODY_JOINT_ORDER):")
    for sdk_idx, j in enumerate(hinge_joints):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        print(f"    sdk[{sdk_idx:2d}] = {name}")

    ChannelFactoryInitialize(args.domain, args.iface)
    buf = LowStateBuffer()
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(buf.update, 10)
    print(f"[SIDECAR] subscribed rt/lowstate on {args.iface} (domain {args.domain}); "
          f"waiting for data...", flush=True)

    metrics = Metrics(args.window, 1.0 / args.hz, args.mode)
    stop = threading.Event()

    # Publish metrics to the MuJoCo sim HUD, and let the sim zero us via DDS.
    try:
        metrics_pub = ChannelPublisher("rt/balance_metrics", String_)
        metrics_pub.Init()
        metrics.publisher = metrics_pub
        print("[SIDECAR] publishing metrics on rt/balance_metrics (sim HUD)", flush=True)
    except Exception as e:
        print(f"[SIDECAR] metrics publish disabled ({e})", flush=True)

    def request_zero(*_):
        metrics.zero()
        print("[SIDECAR] counters zeroed", flush=True)

    def _metrics_cmd(msg):
        if getattr(msg, "data", "").strip().lower() == "zero":
            request_zero()

    try:
        cmd_sub = ChannelSubscriber("rt/metrics_cmd", String_)
        cmd_sub.Init(_metrics_cmd, 1)
    except Exception as e:
        print(f"[SIDECAR] metrics_cmd subscribe disabled ({e})", flush=True)

    signal.signal(signal.SIGUSR1, request_zero)
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def stdin_thread():
        while not stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.25)
            if not r:
                continue
            line = sys.stdin.readline()
            if not line:  # EOF — keep running (piped/backgrounded)
                return
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            cmd = parts[0].lower()
            if cmd == "zero":
                request_zero()
            elif cmd == "mode" and len(parts) > 1:
                metrics.mode_label = parts[1]
                print(f"[SIDECAR] mode label = {parts[1]}", flush=True)
            elif cmd == "note":
                print(f"[NOTE] t={time.monotonic() - metrics.t0:.1f}s "
                      f"{parts[1] if len(parts) > 1 else ''}", flush=True)
            elif cmd == "quit":
                stop.set()
            else:
                print("[SIDECAR] commands: zero | mode <label> | note <text> | quit",
                      flush=True)

    threading.Thread(target=stdin_thread, daemon=True).start()

    ledger = None
    ledger_pub_t = 0.0
    if args.ledger:
        from reward_ledger import RewardLedger
        ledger = RewardLedger(args.ledger)

    dt = 1.0 / args.hz
    warned_stale = False
    started = False
    next_t = time.monotonic()

    while not stop.is_set():
        next_t += dt
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.monotonic()  # fell behind; resync

        msg, stamp = buf.latest()
        now = time.monotonic()
        if msg is None or now - stamp > 0.5:
            if started and not warned_stale:
                print("[SIDECAR] lowstate stale (>0.5s) — metrics paused", flush=True)
                warned_stale = True
            continue
        if not started:
            print("[SIDECAR] lowstate flowing — metrics running", flush=True)
            metrics.zero()  # discard wait time
            started = True
        warned_stale = False

        # --- FK from measured joint angles + IMU orientation ---
        q = [msg.motor_state[i].q for i in range(NUM_JOINTS)]
        quat = list(msg.imu_state.quaternion)  # wxyz
        gyro = np.array(msg.imu_state.gyroscope, dtype=float)
        rpy = np.array(getattr(msg.imu_state, "rpy", (0.0, 0.0, 0.0)), dtype=float)  # firmware Euler (rad)

        for sdk_idx in range(NUM_JOINTS):
            data.qpos[qpos_addr[sdk_idx]] = q[sdk_idx]
        if base_qpos is not None:
            data.qpos[base_qpos:base_qpos + 3] = 0.0
            data.qpos[base_qpos + 3:base_qpos + 7] = quat
        mujoco.mj_kinematics(model, data)

        p_l = data.xpos[bid_l].copy()
        p_r = data.xpos[bid_r].copy()
        dist = float(np.linalg.norm(p_l - p_r))
        low = min(p_l[2], p_r[2])
        rel_h_l = p_l[2] - low
        rel_h_r = p_r[2] - low

        # tilt of gravity axis in base frame (for fall detection)
        R = quat_to_rotmat(quat)
        g_b = R.T @ np.array([0.0, 0.0, -1.0])
        tilt = math.acos(max(-1.0, min(1.0, -g_b[2])))

        metrics.step(dist, rel_h_l, rel_h_r, gyro, tilt, g_b, rpy)

        # --- LIVE reward ledger: publish EVERY tick (50 Hz, operator ask —
        # bars move at policy rate; the sim holds NUMBERS on a 5 Hz snapshot) ---
        if ledger is not None:
            ledger_pub_t = now
            items = ledger.tick(float(g_b[0] ** 2 + g_b[1] ** 2))
            if metrics.publisher is not None and items:
                base_fields = getattr(metrics, "last_payload_fields", "")
                sep = "," if base_fields else ""
                try:
                    metrics.publisher.Write(String_(data=(
                        '{%s%s"ledger":"%s"}' % (base_fields, sep,
                                                 ledger.hud_field(items)))))
                except Exception:
                    pass

    if ledger is not None:
        tape_dir = os.environ.get("LEDGER_TAPE_DIR", ".")
        tape = ledger.save_tape(os.path.join(
            tape_dir, f"reward_tape_mujoco_{time.strftime('%Y%m%d_%H%M%S')}.json"))
        if tape:
            print(f"[LEDGER] tape saved: {tape}", flush=True)
    metrics.summary()


if __name__ == "__main__":
    main()
