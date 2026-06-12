#include "ArmPosePublisher.h"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace h1_2 {

constexpr std::array<float, 14> ArmPosePublisher::DEFAULT_ARM_POS;
constexpr ArmPosePublisher::ModeParams ArmPosePublisher::MODE_PARAMS[6];

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
    trigger_blend();
}

void ArmPosePublisher::set_manual_pose(const std::array<float, 14>& pose) {
    {
        std::lock_guard<std::mutex> lk(manual_mtx_);
        manual_pose_ = pose;
    }
    mode_ = Mode::Manual;
    trigger_blend();
}

void ArmPosePublisher::trigger_blend() {
    // Ease from the last emitted targets to the new pose over transition_s_.
    // Seeded from defaults if nothing was emitted yet.
    blend_start_ = last_targets_;
    blend_t_ = 0.0f;
    blend_active_ = transition_s_ > 0.0f;
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

    // Schedule the next TrainingDist resample (emulated episode reset)
    std::uniform_real_distribution<float> period_u(RESAMPLE_MIN_S, RESAMPLE_MAX_S);
    next_resample_t_ = t_ + period_u(rng_);
}

std::vector<float> ArmPosePublisher::compute_obs_command() const {
    // Manual: the commanded pose IS the obs command (held pose, no wobble).
    if (mode_ == Mode::Manual) {
        std::lock_guard<std::mutex> lk(manual_mtx_);
        return std::vector<float>(manual_pose_.begin(), manual_pose_.end());
    }

    // 14-dim obs: default + held_delta (no wobble in obs — by design)
    std::vector<float> obs(DEFAULT_ARM_POS.begin(), DEFAULT_ARM_POS.end());

    // Apply held deltas to specific joint indices.
    // NOTE (TrainingDist): roll is intentionally UNCLAMPED here — training
    // puts the raw sampled roll in the obs and clamps only the executed
    // target (arm_pose_command.py: command_b vs _update_command).
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

    // TrainingDist: emulate Isaac episode resets — resample the held pose
    // (and wobble phases) every U(10, 15) s. The pose-change blend below
    // turns the resulting step change into a smooth transition.
    if (mode_ == Mode::TrainingDist && t_ >= next_resample_t_) {
        resample_held_pose();
        trigger_blend();
        std::cout << "[ARM_DIST] resample: pitch=" << pitch_delta_
                  << " roll=" << roll_delta_ << " elbow=" << elbow_delta_
                  << " (next in " << (next_resample_t_ - t_) << "s)" << std::endl;
    }

    std::vector<float> targets;

    if (mode_ == Mode::Manual) {
        std::lock_guard<std::mutex> lk(manual_mtx_);
        targets.assign(manual_pose_.begin(), manual_pose_.end());
    } else {
        const auto& p = MODE_PARAMS[static_cast<int>(mode_)];
        const float omega_t = 2.0f * static_cast<float>(M_PI) * p.wobble_freq * t_;

        // Compute wobble for each channel
        const float wob_pitch  = p.wobble_amp * std::sin(omega_t + wobble_phase_[0]);
        const float wob_l_roll = p.wobble_amp * std::sin(omega_t + wobble_phase_[1]);
        const float wob_r_roll = p.wobble_amp * std::sin(omega_t + wobble_phase_[2]);
        const float wob_elbow  = p.wobble_amp * std::sin(omega_t + wobble_phase_[3]);

        // Start from obs (default + held)
        targets = compute_obs_command();

        // Add wobble on top
        targets[IDX_L_SHOULDER_PITCH] += wob_pitch;
        targets[IDX_R_SHOULDER_PITCH] += wob_pitch;
        targets[IDX_L_SHOULDER_ROLL]  += wob_l_roll;
        targets[IDX_R_SHOULDER_ROLL]  += wob_r_roll;
        targets[IDX_L_ELBOW_PITCH]    += wob_elbow;
        targets[IDX_R_ELBOW_PITCH]    += wob_elbow;

        // p7_1b+ self-collision guard: clamp the executed roll OFFSET
        // (held + wobble) to ±0.8, exactly like the training sampler.
        if (mode_ == Mode::TrainingDist) {
            for (size_t idx : {IDX_L_SHOULDER_ROLL, IDX_R_SHOULDER_ROLL}) {
                const float offset = targets[idx] - DEFAULT_ARM_POS[idx];
                targets[idx] = DEFAULT_ARM_POS[idx]
                             + std::clamp(offset, -ROLL_OFFSET_CLAMP, ROLL_OFFSET_CLAMP);
            }
        }
    }

    // Pose-change blend: after any discrete held-pose change (mode switch,
    // TrainingDist resample, stdin `arm` command) the emitted targets ease
    // from the snapshot taken at the change to the new pose over
    // transition_s_ seconds (cosine smoothstep — zero-velocity start/end),
    // so the arm command never jumps. Wobble fades in with the blend.
    if (blend_active_) {
        blend_t_ += dt;
        const float u = std::min(blend_t_ / transition_s_, 1.0f);
        const float s = 0.5f - 0.5f * std::cos(static_cast<float>(M_PI) * u);
        for (size_t i = 0; i < 14; ++i) {
            targets[i] = blend_start_[i] + s * (targets[i] - blend_start_[i]);
        }
        if (u >= 1.0f) blend_active_ = false;
    }

    std::copy(targets.begin(), targets.end(), last_targets_.begin());
    last_targets_valid_ = true;

    return targets;
}

} // namespace h1_2
