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
pose resamples every U(10,15) s to emulate episode resets.

## Arm pose transitions + gains (config.yaml Balance block)

There is ONE commanded held pose. On any discrete change (DPad mode switch,
TrainingDist resample, stdin `arm` command) it cosine-eases (zero-velocity
endpoints) from its current value to the new target over `arm_transition_s`.
Both the policy obs (`arm_pose_command`) AND the arm motor target read this
single slewed pose, so the policy sees the gradual trajectory it must
compensate for, not a step. Wobble (disturbance modes) rides on top of the
slewed pose for the motor target only; the obs stays wobble-free, like training.

- `arm_transition_s` (default 2.0): pose-change slew time, in REAL seconds.
  Timing is measured from the wall clock inside `compute_arm_targets`, so it is
  independent of the loop rate. (Before: the blend was fed `step_dt=0.02` at the
  1 kHz FSM rate, advancing 20× too fast — a "1.5 s" transition completed in
  ~75 ms, i.e. the snap. Fixed.)
- `arm_pose_dwell_s` (optional): TrainingDist resample period. Omit or set ≤0
  for the training-faithful random U(10,15) s; set a value for a fixed dwell so
  each pose fully settles (use longer dwell + slew together).
- `arm_kp` / `arm_kd`: scalar (all 14 arm joints) OR a 14-element list
  (per-joint, 14-dim URDF arm order). Applied to the 14 arm motors only;
  legs+torso keep the deploy.yaml policy-trained gains. Default in config is
  50 / 1.0 (the team's real-robot `BridgeModule/main/config.py` H1_2_KP/H1_2_KD).
  Remove both keys to fall back to deploy.yaml arm gains (kp 100/50, kd 2.0).
  FixStand arm gains in config.yaml are untouched. Per-joint lets you soften
  e.g. only the shoulders (the heaviest CoM lever) while keeping wrists firm.

On startup each RLBase state logs `[FSM] Resolved arm config: arm_transition_s=…
arm_pose_dwell_s=… gain_override=…` plus the resolved kp/kd, so you can confirm
the config took effect (not silently defaulted).

### Tuning for torso stillness (the harness loop)

Hold p8_gold loaded, watch the sidecar `torso_ang_vel RMS` during an arm sweep:
- `arm_kp ∈ {30,50,80}`, `arm_kd ∈ {1.0,1.5,2.0}` (or per-joint — soften
  shoulders first), `arm_transition_s ∈ {1.5,2.5,4.0}`.
- Lowest torso RMS during arm motion while the arm still reaches its pose
  (too-low kp = sag/lag) wins. Config-only — no rebuild between sweep points.

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
