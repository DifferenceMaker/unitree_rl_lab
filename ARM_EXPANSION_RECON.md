# Arm-pose envelope expansion — recon + design (p9 prep)

Branch `aspired/training`. Companion to the p8_gold promotion (Part 1, committed
separately). This documents the current `UniformArmPoseCommand` sampler, the design
for the expanded envelope (yaw, decoupled L/R, overhead pitch, active curriculum
ramp), the proposed self-collision clamp values **flagged for human review**, and
two cross-cutting risks (a reward sign anomaly carried over from gold, and the
warmstart arm-dynamics shift introduced by Part 3).

Files:
- sampler: `source/.../tasks/locomotion/mdp/commands/arm_pose_command.py`
- cfg: `source/.../tasks/locomotion/robots/h1_2/balance_env_cfg.py`
  (`arm_pose_command`, `CurriculumCfg.arm_pose`, `RobotPlayEnvCfg.__post_init__`)
- curriculum fn: `source/.../tasks/locomotion/mdp/curriculums.py:arm_pose_curriculum_phase4`
- arm gains: `source/.../assets/robots/unitree.py` (`UNITREE_H1_2_CFG`, line ~722)

---

## 1. Current sampler internals (gold / p8 behavior)

`UniformArmPoseCommand` (Option-II architecture: the policy commands only legs+torso;
the arms are driven *externally* by this command term, then observed by the policy).

**Sampled DOF (3-DOF subspace, per episode, per env):**
- `pitch_delta`  — **bilateral**: both shoulder_pitch joints get the SAME delta.
- `roll_delta`   — **mirrored**: left_shoulder_roll `+δ`, right_shoulder_roll `−δ`.
- `elbow_delta`  — **bilateral**: both elbow_pitch joints get the SAME delta.
- 4 wobble phase offsets (pitch, left_roll, right_roll, elbow) in `[0, 2π]`.

shoulder_yaw, elbow_roll, wrist_pitch, wrist_yaw are **never sampled** (held at default).

**Per-episode amplitude sampling (v4.1)** — preserved as-is in the expansion:
```
pitch_amp_episode  ~ U(0, pitch_amplitude)     # then delta = U(-1,1)*amp_episode
roll_amp_episode   ~ U(0, roll_amplitude)
elbow_amp_episode  ~ U(0, elbow_amplitude)
wobble_amp_episode ~ U(0, wobble_amplitude)
```
This is why gold covers the whole spectrum each batch: "stand still / no arms"
(amps≈0) → "held pose, no wobble" → "wobble only" → "max disturbance" (amps≈max).

**Held delta → observation (`command_b`, 14-dim):** at reset, `_resample_command`
zeroes `command_b` then writes the held deltas into the obs slots for the matching
joints. `command = default_arm_pos + command_b`. The 14 entries are ALL arm joints
(`ARM_JOINT_REGEX`); only the sampled joints are nonzero, the rest stay 0. The
**wobble is NOT in the obs** — it is a small reactive disturbance the policy is not
told about.

**Held delta + wobble → joint targets:** each step `_update_command` sets joint
position targets = `default + held_delta + wobble_amp_episode*sin(2π·f·t + phase)`.
Only roll is clamped today: `clamp(±roll±wobble, -0.8, 0.8)` (p7_1b self-collision
guard). Pitch and elbow are **unclamped** — they ride into the URDF joint stops.

**Joint→obs index mapping (how new DOF slot in):** `find_joints(REGEX)` returns
articulation-order joint ids; the sampler maps each sub-group to its obs index via
`all_arm_joint_ids.index(joint_id)`. So adding a yaw group follows the exact same
pattern — no hardcoded ordering, no obs-width change. **Obs stays 14-dim**, total
policy obs stays 87 (`3 ang_vel + 3 proj_grav + 27 jpos + 27 jvel + 13 last_action
+ 14 arm_cmd`). Action stays 13 (legs+torso). None of these change.

## 2. Curriculum: why it's inert today, and what "activate" means

`arm_pose_curriculum_phase4` only ramps the **wobble** amplitude over levels; the
held pitch/roll/elbow amplitudes are passed as **constants** and re-asserted onto the
command cfg every step. In the current `balance_env_cfg.py` the wobble levels are
`(0.25, 0.25, 0.25, 0.25)` — all equal — so the curriculum is effectively a no-op
(it just pins the validated p7_1b values). The held envelope (pitch 1.5 / roll 1.0 /
elbow 1.5) never changes.

"Activating the ramp" = making the curriculum grow the **held envelope** (yaw
amplitude, decouple fraction, overhead pitch) across levels, while wobble stays
pinned at 0.25 (validated — do NOT increase). **Level 0 must reproduce gold exactly**
(bilateral pitch 1.5, mirrored roll 1.0, bilateral elbow 1.5, yaw 0, decouple 0,
no overhead) so the gold warmstart sees zero envelope shock at iter 0; the new DOF
ramp UP from there.

## 3. Joint limits (URDF `h1_2.urdf`) — bounds the expansion must respect

| joint | lower | upper | default | notes |
|---|---|---|---|---|
| shoulder_pitch (L/R) | -3.14 | 1.57 | 0.40 | asymmetric; big NEGATIVE headroom → overhead is the −pitch direction (assumed; verify in sim) |
| shoulder_roll L | -0.38 | 3.40 | 0.0 | +δ = abduct/out; inner (toward torso) limited to -0.38 |
| shoulder_roll R | -3.40 | 0.38 | 0.0 | −δ = abduct/out; inner limited to +0.38 |
| shoulder_yaw L | -2.66 | 3.01 | 0.0 | wide |
| shoulder_yaw R | -3.01 | 2.66 | 0.0 | wide |
| elbow_pitch (L/R) | -0.95 | 3.18 | 0.30 | +δ = flex |
| elbow_roll (L/R) | ~-3.0 | ~2.8 | 0.0 | not sampled |
| wrist_pitch (L/R) | -0.47 | 0.47 | 0.0 | tiny |
| wrist_yaw (L/R) | -1.27 | 1.27 | 0.0 | not sampled |

Note gold already commands `pitch = 0.4 ± 1.5 = [-1.1, 1.9]`; the +1.9 exceeds the
+1.57 stop, so gold's pitch was already physically capped on the +side by the joint
limit (no explicit clamp). The expansion adds an **explicit soft-limit clamp** so we
never command past the stops, plus conservative self-collision clamps on top.

## 4. PROPOSED self-collision clamps — ⚠️ FLAG FOR HUMAN REVIEW

Decoupled L/R + yaw + wider/overhead pitch greatly enlarges the self-collision space
(arms into torso/head/each-other). The existing ±0.8 roll clamp is not sufficient.
Proposal — **deliberately conservative; loosen after visual sim check**:

| DOF | proposed clamp (delta from default) | rationale |
|---|---|---|
| shoulder_roll | ±0.8 (keep) | validated p7_1b; inward also limited by URDF |
| shoulder_yaw | ±0.8 (start) | yaw + bent elbow swings forearm toward torso; start tight |
| shoulder_pitch (overhead) | down to default−2.0 (≈ -1.6 cmd) | arm raised but well short of the -3.14 fully-overhead-behind stop where head clipping is likely |
| shoulder_pitch (+side) | soft-limit clamp (≈1.5) | physical stop |
| elbow_pitch | [-0.95, 3.18] soft-limit only | flexion away from body is self-safe |

**In addition, ALL commanded targets are clamped to the articulation soft joint limits
(read at init) with a small margin** — this makes the sampler safe regardless of the
overhead-sign assumption above. The self-collision clamps are tighter bounds layered
on top. All clamp values are cfg fields (see §6) so they can be tuned without code.

## 5. Decouple design (rampable, gold-preserving)

Rather than a hard bilateral/independent toggle, decoupling is a **blend fraction**
`decouple_lr ∈ [0,1]` set by the curriculum:
- sample a shared component and an independent per-arm component;
- `left  = shared + decouple_lr·(left_indep  − shared)`
- `right = mirror(shared) + decouple_lr·(right_indep − mirror(shared))`

At `decouple_lr=0` → exactly gold (bilateral pitch/elbow, mirrored roll). At
`decouple_lr=1` → fully independent arms. The curriculum ramps 0→1, so the warmstart
starts bilateral and gradually learns asymmetry. (The `.md` asked for a `bool` flag;
a float fraction is strictly more general and ramps smoothly — `0.0`/`1.0` recover the
bool endpoints.)

## 6. New cfg knobs (defaults reproduce gold) — for the p9 job files

All added to `UniformArmPoseCommandCfg`, all default to gold-equivalent so the base
task is unchanged until a curriculum/job sets them:

| knob | default | meaning |
|---|---|---|
| `yaw_amplitude` | 0.0 | max \|shoulder_yaw delta\| (U(0,max) per episode) |
| `decouple_lr` | 0.0 | L/R blend: 0 = bilateral/mirrored (gold), 1 = independent |
| `pitch_overhead_amplitude` | 0.0 | extra pitch range in the overhead direction (added to negative side) |
| `pitch_overhead_sign` | -1.0 | which sign is "overhead" (−1 ⇒ negative pitch; flip if sim shows otherwise) |
| `yaw_clamp` | 0.8 | self-collision clamp on yaw target (delta from default) |
| `roll_clamp` | 0.8 | existing ±0.8 roll clamp, now cfg-driven |
| `pitch_overhead_clamp` | 2.0 | max overhead excursion (delta from default) before soft-limit |
| `clamp_to_soft_limits` | True | clamp every target to URDF soft limits ± margin |

Curriculum (`arm_pose_curriculum_phase4`) gets matching per-level tuples that ramp
`yaw_amplitude`, `decouple_lr`, and `pitch_overhead_amplitude` from 0 (level 0 = gold)
up to the full expanded envelope, with wobble pinned at 0.25 throughout.

## 7. ⚠️ Risk flags (do not silently ship)

**(a) torso_stability_bonus sign anomaly (carried over from gold).** The reward fn
returns `exp(-(lin²/std_lin² + ang²/std_ang²))` ∈ (0,1], which is HIGH when the torso
is still. Gold's weight is **−12.0**, so the term *penalizes stillness / rewards torso
motion* — backwards for a "bonus". Part 1 promoted this faithfully (it is what gold
trained on and what was validated in sim2sim/hardware-prep), but it looks like an
accidental sign flip in the gold override JSON. Worth a deliberate A/B (−12 vs +12)
before p9 if there's budget. Documented inline in `balance_env_cfg.py`.

**(b) Warmstart arm-dynamics shift from Part 3.** Gold trained with stiff arms
(shoulder kp 100, yaw/elbow/wrist kp 50, kd 2.0). Part 3 sets the sim arm gains to the
deploy values (kp **40** / kd **3.0**, per the colleague's correction to the original
30/3). This is correct for sim2real fidelity, but it is an **instantaneous dynamics
shift at iter 0** of the p9 warmstart — independent of the command-envelope curriculum.
Softer arms sag more and move slower under the same commanded pose, especially:

**(c) Overhead sag at kp 40.** A shoulder fighting gravity in an overhead pose at
kp 40 may not reach the commanded angle — the arm sags, so the *observed* joint pos
diverges from the commanded held pose in the obs. The policy is told a pose the arm
can't physically hold. Mitigation options for the human: (i) per-joint higher
shoulder_pitch stiffness for overhead, (ii) cap overhead reach lower until gains
settle, (iii) accept the sag as realistic (the real arm sags too). Not resolved here
— surfaced for decision. Part 3 keeps the gains in ONE commented location so this is
a one-line change.
