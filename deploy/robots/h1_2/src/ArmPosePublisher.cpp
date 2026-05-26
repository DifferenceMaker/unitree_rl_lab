#include "ArmPosePublisher.h"

#include <chrono>

namespace h1_2 {

constexpr std::array<float, 14> ArmPosePublisher::DEFAULT_ARM_POS;
constexpr ArmPosePublisher::ModeParams ArmPosePublisher::MODE_PARAMS[4];

ArmPosePublisher& ArmPosePublisher::instance() {
    static ArmPosePublisher inst;
    return inst;
}

ArmPosePublisher::ArmPosePublisher() {
    auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    rng_.seed(static_cast<uint32_t>(seed));
    resample_held_pose();
}

void ArmPosePublisher::set_mode(Mode m) {
    mode_ = m;
    t_ = 0.0f;
    resample_held_pose();
}

void ArmPosePublisher::resample_held_pose() {
    const auto& p = MODE_PARAMS[static_cast<int>(mode_)];

    // Sample held deltas uniformly in [-amp, +amp]
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    pitch_delta_ = u(rng_) * p.pitch_amp;
    roll_delta_  = u(rng_) * p.roll_amp;
    elbow_delta_ = u(rng_) * p.elbow_amp;

    // Sample wobble phases uniformly in [0, 2*pi]
    std::uniform_real_distribution<float> phase_u(0.0f, 2.0f * static_cast<float>(M_PI));
    for (auto& ph : wobble_phase_) ph = phase_u(rng_);
}

std::vector<float> ArmPosePublisher::compute_obs_command() const {
    // 14-dim obs: default + held_delta (no wobble in obs — by design)
    std::vector<float> obs(DEFAULT_ARM_POS.begin(), DEFAULT_ARM_POS.end());

    // Apply held deltas to specific joint indices
    obs[IDX_L_SHOULDER_PITCH] += pitch_delta_;
    obs[IDX_R_SHOULDER_PITCH] += pitch_delta_;
    obs[IDX_L_SHOULDER_ROLL]  += roll_delta_;
    obs[IDX_R_SHOULDER_ROLL]  -= roll_delta_;  // mirrored
    obs[IDX_L_ELBOW_PITCH]    += elbow_delta_;
    obs[IDX_R_ELBOW_PITCH]    += elbow_delta_;

    return obs;
}

std::vector<float> ArmPosePublisher::compute_arm_targets(float dt) {
    // Advance time
    t_ += dt;

    const auto& p = MODE_PARAMS[static_cast<int>(mode_)];
    const float omega_t = 2.0f * static_cast<float>(M_PI) * WOBBLE_FREQ_HZ * t_;

    // Compute wobble for each channel
    const float wob_pitch  = p.wobble_amp * std::sin(omega_t + wobble_phase_[0]);
    const float wob_l_roll = p.wobble_amp * std::sin(omega_t + wobble_phase_[1]);
    const float wob_r_roll = p.wobble_amp * std::sin(omega_t + wobble_phase_[2]);
    const float wob_elbow  = p.wobble_amp * std::sin(omega_t + wobble_phase_[3]);

    // Start from obs (default + held)
    auto targets = compute_obs_command();

    // Add wobble on top
    targets[IDX_L_SHOULDER_PITCH] += wob_pitch;
    targets[IDX_R_SHOULDER_PITCH] += wob_pitch;
    targets[IDX_L_SHOULDER_ROLL]  += wob_l_roll;
    targets[IDX_R_SHOULDER_ROLL]  += wob_r_roll;
    targets[IDX_L_ELBOW_PITCH]    += wob_elbow;
    targets[IDX_R_ELBOW_PITCH]    += wob_elbow;

    return targets;
}

} // namespace h1_2
