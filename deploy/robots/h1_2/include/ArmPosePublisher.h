#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

// Forward declaration of the unitree dds_wrapper rt/arm_sdk subscriber.
// The full type (which pulls in the unitree SDK headers) is only needed in the
// .cpp, so we keep this header light via a unique_ptr<incomplete-type> + an
// out-of-line destructor.
namespace unitree { namespace robot { namespace g1 { namespace subscription { class ArmSdk; } } } }

namespace h1_2 {

/**
 * Singleton publisher for arm pose commands.
 *
 * Mirrors the Python UniformArmPoseCommand class:
 * - Held pose: sampled per episode/mode-switch, applied to specific joints
 * - Wobble:    per-joint sinusoid on top of held pose
 *
 * Mode::Teleop additionally streams a live 14-dim arm pose from xr_teleoperate
 * (published to DDS topic rt/arm_sdk). All teleop behaviour is gated behind
 * Mode::Teleop; every other mode (Idle/Mild/Training/Manipulation) is unchanged.
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
        Teleop = 4,        // live arm pose streamed from xr_teleoperate via rt/arm_sdk
    };

    // SDK motor index for each of the 14 arm joints, in the URDF arm-joint order
    // documented above. Canonical mapping shared with State_RLBase:
    //   element i (publisher layout)  ->  motor_cmd[ARM_SDK_MOTOR_IDS[i]]
    // It doubles as the read map for rt/arm_sdk, because xr_teleoperate publishes
    // each arm joint by the SAME motor index (motor_cmd[13..26]). Reading
    // motor_cmd[ARM_SDK_MOTOR_IDS[i]] therefore lands each value in slot i with
    // no manual reordering.
    static constexpr std::array<int, 14> ARM_SDK_MOTOR_IDS = {
        13, 20, 14, 21, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26
    };

    static ArmPosePublisher& instance();

    ~ArmPosePublisher();  // defined in .cpp (unique_ptr of incomplete type)

    // Called from FSM state entry / DPad. Triggers resample of held pose (non-teleop
    // modes) or (re)arms the teleop bridge + ramp (Mode::Teleop).
    void set_mode(Mode m);

    // Configure the teleop engagement ramp / slew cap. Safe to call any time.
    //   max_speed_rad_s : steady-state per-joint slew cap on the streamed command
    //   ramp_s          : on (re)acquisition the cap eases 0 -> max_speed over this time
    void set_teleop_params(float max_speed_rad_s, float ramp_s);

    // Returns 14-dim obs vector.
    //   non-teleop: default_arm_pos + held_delta (no wobble) — unchanged.
    //   teleop:     the most recently written arm target (1-step consistent).
    std::vector<float> compute_obs_command() const;

    // Returns 14-dim arm joint targets. Advances internal time by dt.
    // Call once per control step.
    //   non-teleop: default + held_delta + wobble — unchanged.
    //   teleop:     slew/ramp-limited stream from rt/arm_sdk (held on timeout).
    std::vector<float> compute_arm_targets(float dt);

    // Torque feedforward (gravity comp) for the 14 arm joints, valid right after the
    // most recent compute_arm_targets() call. Write it to motor_cmd[...].tau() each
    // step. 0 in non-teleop modes / on stale stream.
    const std::array<float, 14>& arm_tau_ff() const { return teleop_tau_; }

    // Force-reset held pose (e.g. on FSM state entry).
    void resample_held_pose();

    Mode mode() const { return mode_; }

private:
    ArmPosePublisher();
    ArmPosePublisher(const ArmPosePublisher&) = delete;
    ArmPosePublisher& operator=(const ArmPosePublisher&) = delete;

    // Lazily creates the rt/arm_sdk subscriber (idempotent). Requires the unitree
    // ChannelFactory to already be initialised (done in main() before the FSM).
    void ensure_teleop_subscriber();

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

    static constexpr ModeParams MODE_PARAMS[5] = {
        /* Idle         */ {0.0f, 0.0f, 0.0f, 0.00f},
        /* Mild         */ {0.5f, 0.3f, 0.5f, 0.05f},
        /* Training     */ {2.5f, 2.0f, 2.5f, 0.50f},
        /* Manipulation */ {0.0f, 0.0f, 0.0f, 0.00f},  // TODO: scripted trajectory
        /* Teleop       */ {0.0f, 0.0f, 0.0f, 0.00f},  // unused: stream comes from rt/arm_sdk
    };

    static constexpr float WOBBLE_FREQ_HZ = 0.05f;

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

    // ---- Teleop bridge state (Mode::Teleop only) ----
    std::unique_ptr<unitree::robot::g1::subscription::ArmSdk> armsdk_sub_;
    // Current (slew/ramp-limited) output; also the obs value. Seeded to default.
    std::array<float, 14> teleop_output_ = DEFAULT_ARM_POS;
    // Gravity-comp torque feedforward read from rt/arm_sdk .tau() (ramp-scaled).
    // 0 when the stream is stale or in non-teleop modes. Written to motor_cmd[].tau().
    std::array<float, 14> teleop_tau_ = {};
    bool  teleop_was_stale_ = true;     // true until a fresh command is acquired
    bool  teleop_warned_stale_ = false; // one-shot "waiting for teleop" log
    float teleop_engage_t_ = 0.0f;      // time since last (re)acquisition, for the ramp
    float teleop_max_speed_ = 6.0f;     // rad/s steady-state slew cap
    float teleop_ramp_s_    = 2.0f;     // ease cap 0 -> max_speed over this many seconds
    static constexpr uint32_t TELEOP_TIMEOUT_MS = 250;  // 50Hz loop, 250Hz publisher
};

} // namespace h1_2
