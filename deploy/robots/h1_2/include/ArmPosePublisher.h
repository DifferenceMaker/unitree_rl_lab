#pragma once

#include <array>
#include <cmath>
#include <mutex>
#include <random>
#include <vector>

namespace h1_2 {

/**
 * Singleton publisher for arm pose commands.
 *
 * Mirrors the Python UniformArmPoseCommand class:
 * - Held pose: sampled per episode/mode-switch, applied to specific joints
 * - Wobble:    per-joint sinusoid on top of held pose, freq=2Hz
 *
 * 14-dim arm joint order (matches URDF resolution from training):
 *   [ 0]  left_shoulder_pitch_joint
 *   [ 1]  right_shoulder_pitch_joint
 *   [ 2]  left_shoulder_roll_joint
 *   [ 3]  right_shoulder_roll_joint
 *   [ 4]  left_shoulder_yaw_joint
 *   [ 5]  right_shoulder_yaw_joint
 *   [ 6]  left_elbow_pitch_joint
 *   [ 7]  right_elbow_pitch_joint
 *   [ 8]  left_elbow_roll_joint
 *   [ 9]  right_elbow_roll_joint
 *   [10]  left_wrist_pitch_joint
 *   [11]  right_wrist_pitch_joint
 *   [12]  left_wrist_yaw_joint
 *   [13]  right_wrist_yaw_joint
 */
class ArmPosePublisher {
public:
    enum class Mode {
        Idle = 0,
        Mild = 1,
        Training = 2,
        Manipulation = 3,
        // Sim2sim harness modes (aspired/deploy_mujoco_harness):
        TrainingDist = 4,  // sampled from the training distribution (p7_1b+ sampler)
        Manual = 5,        // held pose set via stdin `arm <14 vals>` (slew-limited)
    };

    static ArmPosePublisher& instance();

    // Called from FSM state entry. Triggers resample of held pose.
    void set_mode(Mode m);

    // Returns 14-dim obs vector: default_arm_pos + held_delta (no wobble).
    // Matches Python UniformArmPoseCommand.command property.
    // (TrainingDist: held roll is UNCLAMPED here, exactly like training — the
    //  ±0.8 self-collision clamp applies only to the executed targets.)
    std::vector<float> compute_obs_command() const;

    // Returns 14-dim arm joint targets: default + held_delta + wobble.
    // Advances internal time by dt. Call once per control step.
    std::vector<float> compute_arm_targets(float dt);

    // Force-reset held pose (e.g. on FSM state entry).
    void resample_held_pose();

    // Manual mode: set an absolute 14-dim held arm pose (radians, order above).
    // Switches mode to Manual; the executed targets blend toward it.
    void set_manual_pose(const std::array<float, 14>& pose);

    Mode mode() const { return mode_; }

    static constexpr std::array<float, 14> default_arm_pos() { return DEFAULT_ARM_POS; }

    // Pose-change transition time (config.yaml `arm_transition_s`, default 1.5 s).
    void set_transition_s(float s) { transition_s_ = std::max(0.0f, s); }

    // Optional arm kp/kd override (config.yaml `arm_kp`/`arm_kd`; matches the
    // team's real-robot BridgeModule gains, kp=50 kd=1). Negative = no
    // override, deploy.yaml gains stay. Applied by State_RLBase::run().
    void set_arm_gains(float kp, float kd) { arm_kp_ = kp; arm_kd_ = kd; }
    float arm_kp() const { return arm_kp_; }
    float arm_kd() const { return arm_kd_; }

private:
    ArmPosePublisher();
    ArmPosePublisher(const ArmPosePublisher&) = delete;
    ArmPosePublisher& operator=(const ArmPosePublisher&) = delete;

    // 14 arm joint default positions (URDF order)
    // From UNITREE_H1_2_CFG.init_state.joint_pos:
    //   .*_shoulder_pitch_joint: 0.4
    //   .*_elbow_pitch_joint:    0.3
    //   all others:              0.0
    static constexpr std::array<float, 14> DEFAULT_ARM_POS = {
        0.4f, 0.4f,   // shoulder_pitch L, R
        0.0f, 0.0f,   // shoulder_roll  L, R
        0.0f, 0.0f,   // shoulder_yaw   L, R
        0.3f, 0.3f,   // elbow_pitch    L, R
        0.0f, 0.0f,   // elbow_roll     L, R
        0.0f, 0.0f,   // wrist_pitch    L, R
        0.0f, 0.0f,   // wrist_yaw      L, R
    };

    // Per-mode amplitudes
    struct ModeParams {
        float pitch_amp;    // shoulder_pitch hold delta
        float roll_amp;     // shoulder_roll hold delta (mirrored L vs R)
        float elbow_amp;    // elbow_pitch hold delta
        float wobble_amp;   // sinusoidal wobble on top
        float wobble_freq;  // Hz
    };

    // Legacy modes keep their original 0.05 Hz wobble (eyeball-friendly slow
    // sweep). TrainingDist matches the p7_1b+ training sampler exactly:
    // pitch ±1.5, roll ±1.0 (mirrored), elbow ±1.5, wobble 0.25 @ 2 Hz
    // (balance_env_cfg.py arm_pose curriculum + arm_pose_command.py).
    static constexpr ModeParams MODE_PARAMS[6] = {
        /* Idle         */ {0.0f, 0.0f, 0.0f, 0.00f, 0.05f},
        /* Mild         */ {0.5f, 0.3f, 0.5f, 0.05f, 0.05f},
        /* Training     */ {2.5f, 2.0f, 2.5f, 0.50f, 0.05f},
        /* Manipulation */ {0.0f, 0.0f, 0.0f, 0.00f, 0.05f},  // TODO: scripted trajectory
        /* TrainingDist */ {1.5f, 1.0f, 1.5f, 0.25f, 2.00f},
        /* Manual       */ {0.0f, 0.0f, 0.0f, 0.00f, 0.00f},
    };

    // p7_1b+ self-collision guard: executed (held+wobble) shoulder-roll OFFSET
    // from default is clamped to ±0.8 rad. Training applies this at target
    // application time only (arm_pose_command.py _update_command), never in
    // the obs — TrainingDist replicates that exactly.
    static constexpr float ROLL_OFFSET_CLAMP = 0.8f;

    // TrainingDist emulates Isaac episode resets by resampling the held pose
    // every U(RESAMPLE_MIN, RESAMPLE_MAX) seconds.
    static constexpr float RESAMPLE_MIN_S = 10.0f;
    static constexpr float RESAMPLE_MAX_S = 15.0f;

    // Every discrete held-pose change (DPad mode switch, TrainingDist
    // resample, stdin `arm` command) eases over transition_s_ seconds with a
    // cosine blend from the last emitted targets to the new pose, so the arm
    // command never jumps. (Training only changes the held pose at episode
    // reset, where the robot resets with it; mid-run we must ease.)
    static constexpr float DEFAULT_TRANSITION_S = 1.5f;

    // Joint index helpers for the 14-dim layout above
    static constexpr size_t IDX_L_SHOULDER_PITCH = 0;
    static constexpr size_t IDX_R_SHOULDER_PITCH = 1;
    static constexpr size_t IDX_L_SHOULDER_ROLL  = 2;
    static constexpr size_t IDX_R_SHOULDER_ROLL  = 3;
    static constexpr size_t IDX_L_ELBOW_PITCH    = 6;
    static constexpr size_t IDX_R_ELBOW_PITCH    = 7;

    Mode mode_ = Mode::Training;

    // Sampled per episode/mode-switch
    float pitch_delta_ = 0.0f;
    float roll_delta_  = 0.0f;
    float elbow_delta_ = 0.0f;

    // Wobble channels: [0]=pitch, [1]=left_roll, [2]=right_roll, [3]=elbow
    // Each gets an independent phase offset so joints don't wobble in sync.
    std::array<float, 4> wobble_phase_ = {0.0f, 0.0f, 0.0f, 0.0f};

    // Time accumulator (reset on mode change)
    float t_ = 0.0f;

    // TrainingDist resample schedule
    float next_resample_t_ = 0.0f;

    // Manual mode held pose (absolute), guarded for the stdin thread
    std::array<float, 14> manual_pose_ = DEFAULT_ARM_POS;
    mutable std::mutex manual_mtx_;

    // Last targets actually emitted (blend start snapshot; seeded lazily)
    std::array<float, 14> last_targets_ = DEFAULT_ARM_POS;
    bool last_targets_valid_ = false;

    // Pose-change blend state
    void trigger_blend();
    float transition_s_ = DEFAULT_TRANSITION_S;
    std::array<float, 14> blend_start_ = DEFAULT_ARM_POS;
    float blend_t_ = 0.0f;
    bool blend_active_ = false;

    // Arm gain override (negative = use deploy.yaml gains)
    float arm_kp_ = -1.0f;
    float arm_kd_ = -1.0f;

    // RNG
    std::mt19937 rng_;
};

} // namespace h1_2
