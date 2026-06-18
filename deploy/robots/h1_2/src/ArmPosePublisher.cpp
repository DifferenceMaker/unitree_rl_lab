#include "ArmPosePublisher.h"

#include <algorithm>
#include <chrono>
#include <iostream>

#include "unitree/dds_wrapper/robots/g1/g1.h"  // unitree::robot::g1::subscription::ArmSdk

namespace h1_2 {

constexpr std::array<float, 14> ArmPosePublisher::DEFAULT_ARM_POS;
constexpr std::array<int, 14> ArmPosePublisher::ARM_SDK_MOTOR_IDS;
constexpr ArmPosePublisher::ModeParams ArmPosePublisher::MODE_PARAMS[5];

ArmPosePublisher& ArmPosePublisher::instance() {
    static ArmPosePublisher inst;
    return inst;
}

ArmPosePublisher::ArmPosePublisher() {
    auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    rng_.seed(static_cast<uint32_t>(seed));
    resample_held_pose();
}

// Out-of-line: armsdk_sub_ is a unique_ptr to a type only complete in this TU.
ArmPosePublisher::~ArmPosePublisher() = default;

void ArmPosePublisher::set_teleop_params(float max_speed_rad_s, float ramp_s) {
    teleop_max_speed_ = max_speed_rad_s;
    teleop_ramp_s_    = ramp_s;
}

void ArmPosePublisher::ensure_teleop_subscriber() {
    if (armsdk_sub_) return;
    armsdk_sub_ = std::make_unique<unitree::robot::g1::subscription::ArmSdk>();  // "rt/arm_sdk"
    armsdk_sub_->set_timeout_ms(TELEOP_TIMEOUT_MS);
    std::cout << "[ArmPosePublisher] teleop bridge subscribing rt/arm_sdk "
              << "(timeout=" << TELEOP_TIMEOUT_MS << "ms, max_speed=" << teleop_max_speed_
              << " rad/s, ramp=" << teleop_ramp_s_ << "s)" << std::endl;
}

void ArmPosePublisher::set_mode(Mode m) {
    mode_ = m;
    t_ = 0.0f;

    if (m == Mode::Teleop) {
        // Start held at the safe default pose and arm the engagement ramp so the
        // arms ease into the first streamed command instead of snapping to it.
        teleop_output_ = DEFAULT_ARM_POS;
        teleop_was_stale_ = true;
        teleop_warned_stale_ = false;
        teleop_engage_t_ = 0.0f;
        ensure_teleop_subscriber();
    } else {
        resample_held_pose();
    }
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
    // Teleop: the obs is the most recently written arm target (1-step consistent
    // with what the motors are tracking). compute_arm_targets() runs after the
    // policy each step, so this reads the previous step's value — a 20ms lag that
    // matches "the command currently being executed".
    if (mode_ == Mode::Teleop) {
        return std::vector<float>(teleop_output_.begin(), teleop_output_.end());
    }

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
    if (mode_ == Mode::Teleop) {
        // ---- Read the latest streamed command (or detect a stale/absent stream) ----
        std::array<float, 14> live;
        std::array<float, 14> live_tau;
        bool have = false;
        if (armsdk_sub_ && !armsdk_sub_->isTimeout()) {
            std::lock_guard<std::mutex> lk(armsdk_sub_->mutex_);
            const auto& mc = armsdk_sub_->msg_.motor_cmd();
            for (size_t i = 0; i < 14; i++) {
                live[i]     = static_cast<float>(mc[ARM_SDK_MOTOR_IDS[i]].q());
                live_tau[i] = static_cast<float>(mc[ARM_SDK_MOTOR_IDS[i]].tau());  // gravity-comp feedforward
            }
            have = true;
        }

        if (!have) {
            // No / stale stream: hold the last commanded pose (no surprise motion)
            // and re-arm the engagement ramp for the next reacquisition.
            if (!teleop_warned_stale_) {
                std::cout << "[ArmPosePublisher] teleop stream stale — holding last arm pose"
                          << std::endl;
                teleop_warned_stale_ = true;
            }
            teleop_was_stale_ = true;
            teleop_tau_.fill(0.0f);  // no torque feedforward while the stream is stale
            return std::vector<float>(teleop_output_.begin(), teleop_output_.end());
        }

        // Fresh data: on (re)acquisition restart the ramp from the held pose.
        if (teleop_was_stale_) {
            teleop_was_stale_ = false;
            teleop_warned_stale_ = false;
            teleop_engage_t_ = 0.0f;  // ramp_start == current teleop_output_
        }
        teleop_engage_t_ += dt;

        // Engagement ramp: slew cap eases 0 -> max_speed over teleop_ramp_s_, then
        // stays at max_speed. This caps the per-step change so the arms ease into
        // position on engagement (and after any dropout) rather than jumping.
        const float ramp = (teleop_ramp_s_ > 0.0f)
                             ? std::min(teleop_engage_t_ / teleop_ramp_s_, 1.0f)
                             : 1.0f;
        const float max_step = teleop_max_speed_ * ramp * dt;  // max |delta| this step

        for (size_t i = 0; i < 14; i++) {
            float d = live[i] - teleop_output_[i];
            d = std::clamp(d, -max_step, max_step);
            teleop_output_[i] += d;
            // Gravity-comp torque feedforward, eased in over the same engagement ramp
            // so it does not jolt on (re)acquisition. This is the tau xr_teleoperate
            // publishes (IK inverse-dynamics); without it the gravity-loaded joints
            // (shoulder_pitch, elbow) cannot hold position under PD control alone.
            teleop_tau_[i] = ramp * live_tau[i];
            // (Optional OOD safety clamp toward the training range would go here.)
        }
        return std::vector<float>(teleop_output_.begin(), teleop_output_.end());
    }

    // Non-teleop modes are position-only (wobble self-test): no torque feedforward.
    teleop_tau_.fill(0.0f);

    // ---- Non-teleop modes: default + held + wobble (unchanged) ----
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
