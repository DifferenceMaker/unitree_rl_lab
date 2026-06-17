#include "ArmPosePublisher.h"

#include <algorithm>
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
    // Start settled at the default pose (Idle), no transition in progress.
    std::lock_guard<std::mutex> lk(mtx_);
    resample_deltas_locked();
    recompute_held_target_locked();
    slewed_held_ = held_target_;
    blend_active_ = false;
}

// ---- mutating helpers (caller holds mtx_) -------------------------------

void ArmPosePublisher::resample_deltas_locked() {
    const auto& p = MODE_PARAMS[static_cast<int>(mode_)];
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    pitch_delta_ = u(rng_) * p.pitch_amp;
    roll_delta_  = u(rng_) * p.roll_amp;
    elbow_delta_ = u(rng_) * p.elbow_amp;

    std::uniform_real_distribution<float> phase_u(0.0f, 2.0f * static_cast<float>(M_PI));
    for (auto& ph : wobble_phase_) ph = phase_u(rng_);

    // Schedule the next TrainingDist resample.
    float dwell = dwell_s_;
    if (dwell <= 0.0f) {
        std::uniform_real_distribution<float> period_u(RESAMPLE_MIN_S, RESAMPLE_MAX_S);
        dwell = period_u(rng_);
    }
    next_resample_t_ = t_ + dwell;
}

void ArmPosePublisher::recompute_held_target_locked() {
    if (mode_ == Mode::Manual) {
        held_target_ = manual_pose_;
        return;
    }
    held_target_ = DEFAULT_ARM_POS;
    // Roll is left UNCLAMPED here — the obs carries the raw held pose, exactly
    // like training (the ±0.8 clamp applies only to the executed motor target).
    held_target_[IDX_L_SHOULDER_PITCH] += pitch_delta_;
    held_target_[IDX_R_SHOULDER_PITCH] += pitch_delta_;
    held_target_[IDX_L_SHOULDER_ROLL]  += roll_delta_;
    held_target_[IDX_R_SHOULDER_ROLL]  -= roll_delta_;  // mirrored
    held_target_[IDX_L_ELBOW_PITCH]    += elbow_delta_;
    held_target_[IDX_R_ELBOW_PITCH]    += elbow_delta_;
}

void ArmPosePublisher::begin_blend_locked() {
    blend_start_ = slewed_held_;  // ease from where we currently are
    blend_t_ = 0.0f;
    if (transition_s_ > 0.0f) {
        blend_active_ = true;
    } else {
        blend_active_ = false;
        slewed_held_ = held_target_;  // instant
    }
}

float ArmPosePublisher::advance_dt_() {
    auto now = std::chrono::steady_clock::now();
    if (!last_call_valid_) {
        last_call_ = now;
        last_call_valid_ = true;
        return 0.0f;
    }
    float dt = std::chrono::duration<float>(now - last_call_).count();
    last_call_ = now;
    return std::clamp(dt, 0.0f, MAX_DT_S);
}

// ---- public API ----------------------------------------------------------

void ArmPosePublisher::set_mode(Mode m) {
    std::lock_guard<std::mutex> lk(mtx_);
    mode_ = m;
    t_ = 0.0f;
    resample_deltas_locked();
    recompute_held_target_locked();
    begin_blend_locked();
}

void ArmPosePublisher::resample_held_pose() {
    std::lock_guard<std::mutex> lk(mtx_);
    resample_deltas_locked();
    recompute_held_target_locked();
    begin_blend_locked();
}

void ArmPosePublisher::set_manual_pose(const std::array<float, 14>& pose) {
    std::lock_guard<std::mutex> lk(mtx_);
    manual_pose_ = pose;
    mode_ = Mode::Manual;
    recompute_held_target_locked();
    begin_blend_locked();
}

void ArmPosePublisher::set_arm_gains(float kp, float kd) {
    std::lock_guard<std::mutex> lk(mtx_);
    arm_kp_.fill(kp);
    arm_kd_.fill(kd);
    arm_gain_override_ = true;
}

void ArmPosePublisher::set_arm_gains(const std::array<float, 14>& kp,
                                     const std::array<float, 14>& kd) {
    std::lock_guard<std::mutex> lk(mtx_);
    arm_kp_ = kp;
    arm_kd_ = kd;
    arm_gain_override_ = true;
}

std::vector<float> ArmPosePublisher::compute_obs_command() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return std::vector<float>(slewed_held_.begin(), slewed_held_.end());
}

std::vector<float> ArmPosePublisher::compute_arm_targets(float /*dt_hint_ignored*/) {
    std::lock_guard<std::mutex> lk(mtx_);

    const float dt = advance_dt_();  // real elapsed seconds (rate-independent)
    t_ += dt;

    // TrainingDist: emulate Isaac episode resets — resample the held pose and
    // slew to it on the dwell schedule.
    if (mode_ == Mode::TrainingDist && t_ >= next_resample_t_) {
        resample_deltas_locked();
        recompute_held_target_locked();
        begin_blend_locked();
        std::cout << "[ARM_DIST] resample: pitch=" << pitch_delta_
                  << " roll=" << roll_delta_ << " elbow=" << elbow_delta_
                  << " (slew " << transition_s_ << "s)" << std::endl;
    }

    // Advance the slew of the commanded held pose (cosine ease, zero-velocity
    // endpoints). This is the value the obs returns AND the base of the motor
    // target, so the policy sees the same gradual trajectory the arm follows.
    if (blend_active_) {
        blend_t_ += dt;
        const float u = std::min(blend_t_ / transition_s_, 1.0f);
        const float s = 0.5f - 0.5f * std::cos(static_cast<float>(M_PI) * u);
        for (size_t i = 0; i < 14; ++i) {
            slewed_held_[i] = blend_start_[i] + s * (held_target_[i] - blend_start_[i]);
        }
        if (u >= 1.0f) blend_active_ = false;
    }

    // Motor target = slewed held pose + wobble (disturbance modes only).
    std::vector<float> targets(slewed_held_.begin(), slewed_held_.end());

    const auto& p = MODE_PARAMS[static_cast<int>(mode_)];
    if (p.wobble_amp > 0.0f) {
        const float omega_t = 2.0f * static_cast<float>(M_PI) * p.wobble_freq * t_;
        targets[IDX_L_SHOULDER_PITCH] += p.wobble_amp * std::sin(omega_t + wobble_phase_[0]);
        targets[IDX_R_SHOULDER_PITCH] += p.wobble_amp * std::sin(omega_t + wobble_phase_[0]);
        targets[IDX_L_SHOULDER_ROLL]  += p.wobble_amp * std::sin(omega_t + wobble_phase_[1]);
        targets[IDX_R_SHOULDER_ROLL]  += p.wobble_amp * std::sin(omega_t + wobble_phase_[2]);
        targets[IDX_L_ELBOW_PITCH]    += p.wobble_amp * std::sin(omega_t + wobble_phase_[3]);
        targets[IDX_R_ELBOW_PITCH]    += p.wobble_amp * std::sin(omega_t + wobble_phase_[3]);
    }

    // p7_1b+ self-collision guard: clamp executed shoulder-roll OFFSET (held +
    // wobble) to ±0.8 — held roll alone can reach ±1.0, so this applies whether
    // or not wobble is active. Motor target only; obs/slewed_held_ stays
    // unclamped, exactly like the training sampler.
    if (mode_ == Mode::TrainingDist) {
        for (size_t idx : {IDX_L_SHOULDER_ROLL, IDX_R_SHOULDER_ROLL}) {
            const float offset = targets[idx] - DEFAULT_ARM_POS[idx];
            targets[idx] = DEFAULT_ARM_POS[idx]
                         + std::clamp(offset, -ROLL_OFFSET_CLAMP, ROLL_OFFSET_CLAMP);
        }
    }

    return targets;
}

} // namespace h1_2
