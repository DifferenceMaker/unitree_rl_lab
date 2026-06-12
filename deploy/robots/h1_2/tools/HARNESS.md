# H1-2 sim2sim measurement harness

Branch `aspired/deploy_mujoco_harness`. Turns the eyeball MuJoCo sim2sim setup
into a measurement tool. Default behavior of everything is unchanged — all
features are flag/command-gated.

## Components

| Feature | Where | How to use |
|---|---|---|
| Metrics | `tools/balance_metrics.py` (sidecar process) | `python3 balance_metrics.py` (tv env) |
| Training-dist arm mode | controller, `ArmPosePublisher` | DPad **Left** in Balance |
| Scripted pushes | `unitree_mujoco` sim | type `push <vx> <vy>` in the **sim** terminal |
| Manual arm pose | controller | type `arm <14 vals>` / `arm default` in the **controller** terminal |
| EMA action filter | controller, `config.yaml` | `action_ema_alpha: <a>` under `Balance` (default off) |

Sidecar rationale: it reads only `rt/lowstate` (FK from measured joints + IMU,
never sim ground truth), so the production controller stays untouched and the
identical script runs against the real robot on pc4 (`--iface eth0`).

## Metric definitions (match these on the Isaac eval side)

- **touchdowns** — rising edge of foot contact, inferred per foot from FK foot
  height relative to the *lower* foot with hysteresis (lift > 3 cm,
  re-ground < 1.2 cm). Counts are cumulative since start or the last `zero`;
  the printed rate is `total / elapsed-since-zero`.
- **feet_dist** — 3D distance between `left_ankle_roll_link` and
  `right_ankle_roll_link` frames (base-pose invariant). mean/min per window.
- **torso_ang_vel RMS** — `sqrt(mean ‖imu gyro‖²)` per window.
- window = 250 samples @ 50 Hz = 5 s. Falls = gravity tilt > 1.0 rad
  (the controller's `bad_orientation` threshold), rising edge.

Print format:

```
[METRICS] window=250 steps (5.0s) / touchdowns L=3 R=2 total=5 = 0.42/s / feet_dist mean 0.328 min 0.310 / torso_ang_vel RMS 0.142
```

Sidecar stdin: `zero` (also SIGUSR1) · `mode <label>` · `note <text>` · `quit`.
On exit it prints a RUN SUMMARY (duration, mode label, touchdowns, falls,
mean feet_dist, overall ang-vel RMS).

## Training-distribution arm mode (DPad Left)

Replicates the p7_1b+ training sampler exactly (`arm_pose_command.py` +
`balance_env_cfg.py` arm_pose curriculum): held pose pitch ±1.5 / roll ±1.0
mirrored / elbow ±1.5, wobble 0.25 @ 2 Hz with per-channel phases, executed
shoulder-roll offset clamped to ±0.8 (obs unclamped, like training). The held
pose resamples every U(10,15) s to emulate episode resets; transitions are
slew-limited (3 rad/s) so the resample can't step the targets. Legacy modes
(Up/Right/Down) are byte-for-byte unchanged.

## p7_1c vs p7_1d comparison runbook (training-dist mode)

Three terminals (suffix each log with the policy name):

```bash
# T1 — sim
cd ~/Projects/robot_projects/repos/unitree_mujoco/simulate/build
./unitree_mujoco | tee /tmp/sim_$(date +%H%M%S).log

# T2 — controller (set policy_dir in config/config.yaml first; verify
#      action_ema_alpha is absent/0 for comparison runs)
cd ~/Projects/robot_projects/repos/unitree_rl_lab/deploy/robots/h1_2/build
./h1_2_ctrl --network lo 2>&1 | tee /tmp/ctrl_p7_1c.log

# T3 — metrics sidecar
conda activate tv
cd ~/Projects/robot_projects/repos/unitree_rl_lab/deploy/robots/h1_2/tools
python3 -u balance_metrics.py --mode p7_1c_trainingdist | tee /tmp/metrics_p7_1c.log
```

Per policy:
1. Gamepad: `LT+Up` FixStand → `RB+B` Balance. Robot settles ~5 s.
2. DPad **Left** → controller prints `[ARM_MODE] -> TRAINING-DIST`.
3. In T3 type `zero` (discard the settling transient). Start the clock.
4. Run ≥ 300 s untouched (≥ 20 resamples). Optionally add scripted pushes —
   same schedule for both policies, e.g. `push 0.3 0` in T1 at 60/120/180 s
   (the `[PUSH]` log lines give exact timestamps for the rerun).
5. Ctrl+C the sidecar → RUN SUMMARY. Swap `policy_dir` to p7_1d in
   config.yaml, restart the controller (sim can stay up), repeat 1-5.

Read-out: lower touchdown rate = fewer counter-steps (stiller stance); higher
feet_dist min = less foot-crossing; lower torso RMS = stabler camera platform.
Falls trump everything. Compare the post-`zero` [METRICS] windows
side-by-side and the two RUN SUMMARYs; the same definitions can be computed
in the Isaac eval for a 3-way sim2sim2train comparison.

## EMA filter tuning (NOT for comparison runs)

Uncomment `action_ema_alpha` in `config/config.yaml` (try 0.6-0.8, NeRF2Real
used 0.8), rebuild not needed (config is read at startup). The controller
prints a loud warning when the filter is active. The `last_action` obs always
carries the raw policy output, so the obs contract is intact; only the
executed targets are smoothed.

## Build

```bash
cd ~/Projects/robot_projects/repos/unitree_rl_lab/deploy/robots/h1_2/build && cmake .. && make -j4
cd ~/Projects/robot_projects/repos/unitree_mujoco/simulate/build && cmake .. && make -j4
```
