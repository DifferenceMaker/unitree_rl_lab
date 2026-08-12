#!/usr/bin/env python3
"""walk_teleop.py — keyboard velocity teleop for the Walk sim2sim state.

Publishes unitree_go::msg::dds_::WirelessController_ on rt/wirelesscontroller
(lo, domain 0) at 50 Hz — the SAME topic the real remote publishes on hardware,
so the controller-side wiring (WirelessJoystick -> velocity_commands obs) is
identical in sim and on the robot.

Runs in its OWN terminal tab — the MuJoCo window's arrow keys stay with the
elastic band; there is no clash because keystrokes only reach the focused
window.

Keys (velocity_commands mapping: obs = [ly, -lx, -rx]):
    Up/Down     vx forward/backward   (ly)
    Left/Right  vy strafe left/right  (-lx)
    Q / E       yaw left/right        (-rx)
    Space       ZERO everything now
    Tab         sticky mode toggle: OFF = release decays to 0 in ~0.5 s
                                     ON  = command HOLDS until changed
    Esc / Ctrl-C  quit (publishes zeros on exit)

Ramped, not stepped: holding a key accumulates toward the clamp; feel is a
stick, not a switch. Clamps mirror the lm2 deploy contract:
    vx [-0.3, 1.0]   vy [-0.3, 0.3]   wz [-0.5, 0.5]
"""
import curses
import time
import sys

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

RATE_HZ = 50.0
DT = 1.0 / RATE_HZ
RAMP = 1.2          # units/s toward target while key held
DECAY = 2.0         # units/s toward 0 when released (non-sticky), ~0.5 s from 1.0
CLAMPS = {"vx": (-0.3, 1.0), "vy": (-0.3, 0.3), "wz": (-0.5, 0.5)}
HOLD_S = 0.15       # a key is "held" while its last press is younger than this


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main(scr):
    curses.cbreak()
    scr.nodelay(True)
    scr.keypad(True)

    ChannelFactoryInitialize(0, "lo")
    pub = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
    pub.Init()

    vx = vy = wz = 0.0
    sticky = False
    last = {"vx+": 0.0, "vx-": 0.0, "vy+": 0.0, "vy-": 0.0, "wz+": 0.0, "wz-": 0.0}

    def held(k, now):
        return (now - last[k]) < HOLD_S

    while True:
        now = time.monotonic()
        # ---- keys ----
        while True:
            c = scr.getch()
            if c == -1:
                break
            if c == curses.KEY_UP:      last["vx+"] = now
            elif c == curses.KEY_DOWN:  last["vx-"] = now
            elif c == curses.KEY_LEFT:  last["vy+"] = now
            elif c == curses.KEY_RIGHT: last["vy-"] = now
            elif c in (ord("q"), ord("Q")): last["wz+"] = now
            elif c in (ord("e"), ord("E")): last["wz-"] = now
            elif c == ord(" "):
                vx = vy = wz = 0.0
            elif c == ord("\t"):
                sticky = not sticky
            elif c == 27:  # Esc
                raise KeyboardInterrupt

        # ---- ramp / decay ----
        def step(v, kp, km, name):
            up, dn = held(kp, now), held(km, now)
            if up and not dn:
                v += RAMP * DT
            elif dn and not up:
                v -= RAMP * DT
            elif not sticky:
                d = DECAY * DT
                v = 0.0 if abs(v) <= d else v - d * (1 if v > 0 else -1)
            lo, hi = CLAMPS[name]
            return clamp(v, lo, hi)

        vx = step(vx, "vx+", "vx-", "vx")
        vy = step(vy, "vy+", "vy-", "vy")
        wz = step(wz, "wz+", "wz-", "wz")

        # ---- publish (velocity_commands reads: vx=ly, vy=-lx, wz=-rx) ----
        msg = WirelessController_(lx=-vy, ly=vx, rx=-wz, ry=0.0, keys=0)
        pub.Write(msg)

        # ---- HUD ----
        scr.erase()
        scr.addstr(0, 0, "WALK TELEOP  (rt/wirelesscontroller @ 50 Hz, lo/domain 0)")
        scr.addstr(2, 0, f"  vx {vx:+.2f} m/s   [{CLAMPS['vx'][0]}, {CLAMPS['vx'][1]}]")
        scr.addstr(3, 0, f"  vy {vy:+.2f} m/s   [+left/-right]")
        scr.addstr(4, 0, f"  wz {wz:+.2f} rad/s [Q left / E right]")
        scr.addstr(6, 0, f"  mode: {'STICKY (Tab to release)' if sticky else 'decay (Tab for sticky)'}")
        scr.addstr(7, 0, "  Space = zero   Esc = quit")
        bar = int((vx - CLAMPS["vx"][0]) / (CLAMPS["vx"][1] - CLAMPS["vx"][0]) * 40)
        scr.addstr(9, 0, "  vx [" + "#" * bar + "-" * (40 - bar) + "]")
        scr.refresh()

        time.sleep(DT)


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    finally:
        # zeros on exit so the policy never keeps a stale command
        try:
            ChannelFactoryInitialize(0, "lo")
            pub = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
            pub.Init()
            for _ in range(5):
                pub.Write(WirelessController_(lx=0.0, ly=0.0, rx=0.0, ry=0.0, keys=0))
                time.sleep(0.02)
        except Exception:
            pass
        print("walk_teleop: exited, zeros published.")
