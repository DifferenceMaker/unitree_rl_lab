#pragma once

#include <array>
#include <cmath>
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
    };

    static ArmPosePublisher& instance();

    // Called from FSM state entry. Triggers resample of held pose.
    void set_mode(Mode m);

    // Returns 14-dim obs vector: default_arm_pos + held_delta (no wobble).
    // Matches Python UniformArmPoseCommand.command property.
    std::vector<float> compute_obs_command() const;

    // Returns 14-dim arm joint targets: default + held_delta + wobble.
    // Advances internal time by dt. Call once per control step.
    std::vector<float> compute_arm_targets(float dt);

    // Force-reset held pose (e.g. on FSM state entry).
    void resample_held_pose();

    Mode mode() const { return mode_; }

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
        float pitch_amp;   // shoulder_pitch hold delta
        float roll_amp;    // shoulder_roll hold delta (mirrored L vs R)
        float elbow_amp;   // elbow_pitch hold delta
        float wobble_amp;  // sinusoidal wobble on top
    };

    static constexpr ModeParams MODE_PARAMS[4] = {
        /* Idle         */ {0.0f, 0.0f, 0.0f, 0.00f},
        /* Mild         */ {0.5f, 0.3f, 0.5f, 0.05f},
        /* Training     */ {1.5f, 1.0f, 1.5f, 0.15f},
        /* Manipulation */ {0.0f, 0.0f, 0.0f, 0.00f},  // TODO: scripted trajectory
    };

    static constexpr float WOBBLE_FREQ_HZ = 2.0f;

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

    // RNG
    std::mt19937 rng_;
};

} // namespace h1_2
