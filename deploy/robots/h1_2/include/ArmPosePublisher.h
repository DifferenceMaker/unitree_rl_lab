#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <mutex>
#include <random>
#include <vector>

namespace h1_2 {

/**
 * Singleton publisher for arm pose commands (13-action balance policies: the
 * policy drives legs+torso, the arms are commanded externally from here).
 *
 * Design (see HARNESS.md / session notes):
 * - There is ONE "commanded held pose" (14-dim) that the policy obs
 *   (arm_pose_command) and the arm motor targets BOTH derive from. When the
 *   held pose changes (DPad mode switch, TrainingDist resample, stdin `arm`
 *   command) the commanded pose SLEWS from where it was to the new target over
 *   `arm_transition_s` seconds (cosine ease). The obs sees the in-progress
 *   slewed pose, not the final target, so the policy gets a smooth feedforward
 *   trajectory to anticipate the CoM shift rather than a step it can only react
 *   to late. Wobble (disturbance modes) rides on top of the slewed pose for the
 *   MOTOR target only; the obs is wobble-free, matching training.
 * - Timing is measured from the wall clock inside compute_arm_targets(), so the
 *   slew/wobble advance in real seconds regardless of how fast the caller ticks
 *   (the FSM loop runs at 1 kHz but step_dt is 0.02 — passing step_dt advanced
 *   time 20x too fast, which made a configured 1.5 s transition complete in
 *   ~75 ms: the "snap").
 *
 * 14-dim arm joint order (matches URDF resolution from training):
 *   [ 0] left_shoulder_pitch   [ 1] right_shoulder_pitch
 *   [ 2] left_shoulder_roll    [ 3] right_shoulder_roll
 *   [ 4] left_shoulder_yaw     [ 5] right_shoulder_yaw
 *   [ 6] left_elbow_pitch      [ 7] right_elbow_pitch
 *   [ 8] left_elbow_roll       [ 9] right_elbow_roll
 *   [10] left_wrist_pitch      [11] right_wrist_pitch
 *   [12] left_wrist_yaw        [13] right_wrist_yaw
 */
class ArmPosePublisher {
public:
    enum class Mode {
        Idle = 0,
        Mild = 1,
        Training = 2,
        Manipulation = 3,
        // Sim2sim harness modes:
        TrainingDist = 4,  // sampled from the training distribution (p7_1b+ sampler)
        Manual = 5,        // held pose set via stdin `arm <14 vals>`
    };

    static ArmPosePublisher& instance();

    // FSM state entry / DPad: switch mode, resample the held pose, start a slew.
    void set_mode(Mode m);

    // 14-dim policy obs: the CURRENT slewed commanded held pose (no wobble).
    // Reflects the in-progress transition, not the final target.
    std::vector<float> compute_obs_command() const;

    // 14-dim arm motor targets: slewed commanded held pose + wobble (+ roll
    // clamp for TrainingDist). Advances slew/wobble by the real elapsed time
    // since the previous call (the float arg is ignored — see class comment).
    std::vector<float> compute_arm_targets(float dt_hint_ignored);

    // Force-resample the held pose for the current mode and start a slew to it.
    void resample_held_pose();

    // Manual mode: set an absolute 14-dim held arm pose (radians, order above).
    // Switches to Manual; the commanded pose slews there over arm_transition_s.
    void set_manual_pose(const std::array<float, 14>& pose);

    Mode mode() const { return mode_; }

    static constexpr std::array<float, 14> default_arm_pos() { return DEFAULT_ARM_POS; }

    // --- Per-state config (set by State_RLBase from config.yaml) ---

    // Pose-change transition time (`arm_transition_s`, default below).
    void set_transition_s(float s) { transition_s_ = std::max(0.0f, s); }
    float transition_s() const { return transition_s_; }

    // TrainingDist resample dwell (`arm_pose_dwell_s`). > 0 = fixed period;
    // <= 0 (default) = emulate Isaac episode resets with U(10, 15) s.
    void set_dwell_s(float s) { dwell_s_ = s; }
    float dwell_s() const { return dwell_s_; }

    // Arm kp/kd override applied to the 14 arm motors only (`arm_kp`/`arm_kd`,
    // scalar or per-joint). No override => deploy.yaml arm gains stay. Applied
    // every step by State_RLBase::run().
    void set_arm_gains(float kp, float kd);
    void set_arm_gains(const std::array<float, 14>& kp, const std::array<float, 14>& kd);
    bool arm_gain_override() const { return arm_gain_override_; }
    float arm_kp(size_t i) const { return arm_kp_[i]; }
    float arm_kd(size_t i) const { return arm_kd_[i]; }

private:
    ArmPosePublisher();
    ArmPosePublisher(const ArmPosePublisher&) = delete;
    ArmPosePublisher& operator=(const ArmPosePublisher&) = delete;

    // 14 arm joint default positions (URDF order). From UNITREE_H1_2_CFG:
    //   shoulder_pitch 0.4, elbow_pitch 0.3, all others 0.0
    static constexpr std::array<float, 14> DEFAULT_ARM_POS = {
        0.4f, 0.4f,   // shoulder_pitch L, R
        0.0f, 0.0f,   // shoulder_roll  L, R
        0.0f, 0.0f,   // shoulder_yaw   L, R
        0.3f, 0.3f,   // elbow_pitch    L, R
        0.0f, 0.0f,   // elbow_roll     L, R
        0.0f, 0.0f,   // wrist_pitch    L, R
        0.0f, 0.0f,   // wrist_yaw      L, R
    };

    struct ModeParams {
        float pitch_amp;    // shoulder_pitch hold delta
        float roll_amp;     // shoulder_roll hold delta (mirrored L vs R)
        float elbow_amp;    // elbow_pitch hold delta
        float wobble_amp;   // sinusoidal wobble on top (motor target only)
        float wobble_freq;  // Hz
    };

    // Legacy modes keep their slow 0.05 Hz wobble. TrainingDist matches the
    // p7_1b+ training sampler (pitch ±1.5, roll ±1.0 mirrored, elbow ±1.5,
    // wobble 0.25 @ 2 Hz).
    static constexpr ModeParams MODE_PARAMS[6] = {
        /* Idle         */ {0.0f, 0.0f, 0.0f, 0.00f, 0.05f},
        /* Mild         */ {0.5f, 0.3f, 0.5f, 0.05f, 0.05f},
        /* Training     */ {2.5f, 2.0f, 2.5f, 0.50f, 0.05f},
        /* Manipulation */ {0.0f, 0.0f, 0.0f, 0.00f, 0.05f},
        /* TrainingDist */ {1.5f, 1.0f, 1.5f, 0.25f, 2.00f},
        /* Manual       */ {0.0f, 0.0f, 0.0f, 0.00f, 0.00f},
    };

    // p7_1b+ self-collision guard: executed shoulder-roll OFFSET clamped to
    // ±0.8 (motor target only; obs is unclamped, like training).
    static constexpr float ROLL_OFFSET_CLAMP = 0.8f;

    static constexpr float RESAMPLE_MIN_S = 10.0f;
    static constexpr float RESAMPLE_MAX_S = 15.0f;
    static constexpr float DEFAULT_TRANSITION_S = 2.0f;
    static constexpr float MAX_DT_S = 0.05f;  // clamp wall-clock dt against stalls

    static constexpr size_t IDX_L_SHOULDER_PITCH = 0;
    static constexpr size_t IDX_R_SHOULDER_PITCH = 1;
    static constexpr size_t IDX_L_SHOULDER_ROLL  = 2;
    static constexpr size_t IDX_R_SHOULDER_ROLL  = 3;
    static constexpr size_t IDX_L_ELBOW_PITCH    = 6;
    static constexpr size_t IDX_R_ELBOW_PITCH    = 7;

    // Sample held deltas + wobble phases for the current mode, and schedule the
    // next TrainingDist resample. Caller holds mtx_.
    void resample_deltas_locked();
    // Compute the destination held pose (default + held_delta, roll UNCLAMPED)
    // for the current mode/deltas into held_target_. Caller holds mtx_.
    void recompute_held_target_locked();
    // Snapshot the current slewed pose and start a transition to held_target_.
    void begin_blend_locked();
    float advance_dt_();  // real elapsed seconds since last call, clamped

    Mode mode_ = Mode::Idle;

    // Held-pose deltas (sampled per mode switch / resample)
    float pitch_delta_ = 0.0f;
    float roll_delta_  = 0.0f;
    float elbow_delta_ = 0.0f;

    // Wobble: per-channel phase [pitch, left_roll, right_roll, elbow]
    std::array<float, 4> wobble_phase_ = {0.0f, 0.0f, 0.0f, 0.0f};
    float t_ = 0.0f;                 // accumulated real time (wobble clock)
    float next_resample_t_ = 0.0f;   // TrainingDist resample schedule

    // Manual held pose (absolute)
    std::array<float, 14> manual_pose_ = DEFAULT_ARM_POS;

    // Slew state: commanded held pose ramps blend_start_ -> held_target_.
    std::array<float, 14> held_target_  = DEFAULT_ARM_POS;  // destination held pose
    std::array<float, 14> blend_start_  = DEFAULT_ARM_POS;  // pose at last change
    std::array<float, 14> slewed_held_  = DEFAULT_ARM_POS;  // current (obs reads this)
    float transition_s_ = DEFAULT_TRANSITION_S;
    float blend_t_ = 0.0f;
    bool  blend_active_ = false;

    float dwell_s_ = -1.0f;  // <=0 => random U(10,15)

    // Arm gain override (per-joint)
    bool arm_gain_override_ = false;
    std::array<float, 14> arm_kp_ = {};
    std::array<float, 14> arm_kd_ = {};

    // Wall-clock timing
    std::chrono::steady_clock::time_point last_call_;
    bool last_call_valid_ = false;

    // Guards all mutable slew/mode/manual state (producer = compute_arm_targets
    // on the 1 kHz FSM thread; readers = obs on the 50 Hz policy thread and the
    // stdin thread via set_manual_pose).
    mutable std::mutex mtx_;

    std::mt19937 rng_;
};

} // namespace h1_2
