#include <iostream>
#include <cmath>
#include <algorithm>
#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "ArmPosePublisher.h"
#include "PolicyStatusPublisher.h"

// Arm SDK motor indices in URDF arm joint order (14 entries).
// Derived from joint_ids_map[articulation_idx] for each arm joint.
// Matches standard Unitree HG protocol H1-2 motor numbering:
//   13=L_SHOULDER_PITCH, 20=R_SHOULDER_PITCH, 14=L_SHOULDER_ROLL, etc.
static constexpr int ARM_SDK_MOTOR_IDS[14] = {
    13, 20, 14, 21, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26
};

// URDF (= articulation) indices of the 13 leg+torso joints, in policy action order.
// Matches actions.JointPositionAction.joint_ids in deploy.yaml.
// Used to route action[i] → joint_ids_map[LEG_TORSO_URDF_IDS[i]] → SDK motor index.
static constexpr int LEG_TORSO_URDF_IDS[13] = {
    0,   // action[0]  L_HIP_YAW
    1,   // action[1]  R_HIP_YAW
    2,   // action[2]  TORSO
    3,   // action[3]  L_HIP_PITCH
    4,   // action[4]  R_HIP_PITCH
    7,   // action[5]  L_HIP_ROLL
    8,   // action[6]  R_HIP_ROLL
    11,  // action[7]  L_KNEE
    12,  // action[8]  R_KNEE
    15,  // action[9]  L_ANK_PITCH
    16,  // action[10] R_ANK_PITCH
    19,  // action[11] L_ANK_ROLL
    20,  // action[12] R_ANK_ROLL
};

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // Optional EMA action filter — sim2sim filter tuning ONLY. Configured in
    // the controller's config.yaml (Balance block), never deploy.yaml. MUST
    // stay 0.0 / absent for training-comparison runs.
    if (cfg["action_ema_alpha"]) {
        env->action_ema_alpha = cfg["action_ema_alpha"].as<float>();
        if (env->action_ema_alpha > 0.0f) {
            std::cout << "[FSM]   *** EMA ACTION FILTER ON (alpha="
                      << env->action_ema_alpha
                      << ") — NOT comparable to training/Isaac eval ***" << std::endl;
        }
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );

    // Phase 5 v3 is Option II: 13-action policy controlling only legs+torso.
    // Arms driven externally by ArmPosePublisher. Initialize to Training mode
    // (matches training distribution: held pose ±1.5/±1.0/±1.5, wobble 0.15).
    auto& arm_pub = h1_2::ArmPosePublisher::instance();

    // --- Per-state arm config (config.yaml; applied to the arm path only) ---
    // Set transition/dwell BEFORE set_mode so the first blend + resample
    // schedule pick them up.
    if (cfg["arm_transition_s"]) {
        arm_pub.set_transition_s(cfg["arm_transition_s"].as<float>());
    }
    if (cfg["arm_pose_dwell_s"]) {
        arm_pub.set_dwell_s(cfg["arm_pose_dwell_s"].as<float>());
    }

    // Arm kp/kd override for the 14 arm motors only (legs+torso keep the
    // policy-trained deploy.yaml gains). Each key may be a scalar (all arm
    // joints) OR a 14-element list (per-joint, 14-dim URDF arm order — see
    // ArmPosePublisher.h). Absent => deploy.yaml arm gains apply.
    if (cfg["arm_kp"] && cfg["arm_kd"]) {
        auto parse_gain = [](const YAML::Node& n, std::array<float, 14>& out) -> int {
            if (n.IsSequence()) {
                if (n.size() != 14) return -1;       // bad length
                for (size_t i = 0; i < 14; ++i) out[i] = n[i].as<float>();
                return 14;                            // per-joint
            }
            out.fill(n.as<float>());
            return 0;                                 // scalar
        };
        std::array<float, 14> kp{}, kd{};
        int rp = parse_gain(cfg["arm_kp"], kp);
        int rd = parse_gain(cfg["arm_kd"], kd);
        if (rp < 0 || rd < 0) {
            std::cout << "[FSM]   WARNING: arm_kp/arm_kd must be a scalar or a 14-element "
                         "list — ignoring, deploy.yaml arm gains kept." << std::endl;
        } else {
            arm_pub.set_arm_gains(kp, kd);
            std::cout << "[FSM]   Arm gain override (14 arm motors only):" << std::endl;
            if (rp == 0 && rd == 0) {
                std::cout << "[FSM]     kp=" << kp[0] << " kd=" << kd[0] << " (scalar)" << std::endl;
            } else {
                std::cout << "[FSM]     kp=[";
                for (size_t i = 0; i < 14; ++i) std::cout << kp[i] << (i < 13 ? "," : "");
                std::cout << "]" << std::endl << "[FSM]     kd=[";
                for (size_t i = 0; i < 14; ++i) std::cout << kd[i] << (i < 13 ? "," : "");
                std::cout << "]" << std::endl;
            }
        }
    }

    // Default arm source: IDLE (held at default pose, no motion).
    arm_pub.set_mode(h1_2::ArmPosePublisher::Mode::Idle);

    std::cout << "[FSM] State_RLBase " << state_string << " constructed." << std::endl;
    std::cout << "[FSM]   Resolved arm config: arm_transition_s=" << arm_pub.transition_s()
              << "  arm_pose_dwell_s=" << arm_pub.dwell_s()
              << " (<=0 => random 10-15s)"
              << "  gain_override=" << (arm_pub.arm_gain_override() ? "yes" : "no (deploy.yaml)")
              << std::endl;
    std::cout << "[FSM]   Arm mode default: IDLE (no motion). DPad to change:" << std::endl;
    std::cout << "[FSM]   Up=Idle | Right=Mild | Down=SafeDist (gentle, real-robot-safe) | Left=TrainingDist (full, sim2sim)" << std::endl;
    std::cout << "[FSM]   Every pose change slews over arm_transition_s. "
                 "stdin: `arm <14 vals>` sets a manual held pose." << std::endl;
}

void State_RLBase::run()
{
    // ===== Arm mode switching via DPad (rising-edge triggered) =====
    // Up=Idle, Right=Mild, Down=SafeDist (gentle, real-robot-safe), Left=TrainingDist (full, sim2sim).
    // Operates independent of FSM state — works in any BalancePush variant.
    using ArmMode = h1_2::ArmPosePublisher::Mode;
    auto& arm_pub = h1_2::ArmPosePublisher::instance();  // mode switches blend over arm_transition_s

    // Sim2sim HUD: report the active policy + arm gains on rt/policy_status (throttled).
    h1_2::PolicyStatusPublisher::instance().publish(getStateString());

    if (lowstate->joystick.up.on_pressed) {
        arm_pub.set_mode(ArmMode::Idle);
        std::cout << "[ARM_MODE] -> IDLE (no motion)" << std::endl;
    }
    if (lowstate->joystick.right.on_pressed) {
        arm_pub.set_mode(ArmMode::Mild);
        std::cout << "[ARM_MODE] -> MILD (gentle disturbance)" << std::endl;
    }
    if (lowstate->joystick.down.on_pressed) {
        arm_pub.set_mode(ArmMode::SafeDist);
        std::cout << "[ARM_MODE] -> SAFE-DIST (gentle sampled, real-robot-safe:"
                  << " pitch±0.6 roll±0.3 elbow±0.6, +0.45rad raise (arms stay forward),"
                  << " wobble 0.05@0.05Hz, resample 20-30s)" << std::endl;
    }
    // (Down was full Training ±2.5 OOD — now SafeDist above. Full TrainingDist stays
    //  on Left for sim2sim stress; Mode::Training remains in the enum, dpad-unreachable.)
    if (lowstate->joystick.left.on_pressed) {
        arm_pub.set_mode(ArmMode::TrainingDist);
        std::cout << "[ARM_MODE] -> TRAINING-DIST (sampled: pitch±1.5 roll±1.0 elbow±1.5,"
                  << " +0.45rad shoulder-pitch raise (anti-self-collision),"
                  << " wobble 0.25@2Hz, roll clamp ±0.8, resample 10-15s)" << std::endl;
    }


    auto action = env->action_manager->processed_actions();

    // DEBUG: log action + obs values once per second
    static int log_counter = 0;
    if (log_counter++ % 50 == 0) {
        std::cout << "[RL] action[0..2]=" << action[0] << "," << action[1] << "," << action[2];

        try {
            auto obs_map = env->observation_manager->compute();
            for (const auto& kv : obs_map) {
                const auto& flat = kv.second;
                std::cout << " obs[" << kv.first << "].size=" << flat.size()
                          << " [0..3]=" << flat[0] << "," << flat[1] << "," << flat[2];
                int nan_count = 0;
                for (auto v : flat) if (std::isnan(v)) nan_count++;
                std::cout << " nan_count=" << nan_count;
            }
        } catch (std::exception& e) {
            std::cout << " obs_err=" << e.what();
        }

        std::cout << " current_q[0]=" << lowstate->msg_.motor_state()[0].q();
        std::cout << std::endl;
    }
    if (log_counter == 50) {  // print once after first second
        std::cout << "[DIAG] Action write map:" << std::endl;
        for (size_t i = 0; i < 13; i++) {
            int urdf_idx = LEG_TORSO_URDF_IDS[i];
            int motor_idx = env->robot->data.joint_ids_map[urdf_idx];
            std::cout << "  action[" << i << "]=" << action[i]
                    << " → URDF " << urdf_idx
                    << " → motor " << motor_idx << std::endl;
        }
        std::cout << "[DIAG] Arm write map:" << std::endl;
        auto arm_now = h1_2::ArmPosePublisher::instance().compute_obs_command();
        for (size_t i = 0; i < 14; i++) {
            std::cout << "  arm[" << i << "]=" << arm_now[i]
                    << " → motor " << ARM_SDK_MOTOR_IDS[i] << std::endl;
        }
    }

    if (action.size() == 13) {
        // 13-action policy (Option II / v3): action is in policy action order, indexed
        // via LEG_TORSO_URDF_IDS to get the URDF (= articulation) index, then via
        // joint_ids_map to get the SDK motor index.
        for (size_t i = 0; i < 13; i++) {
            int urdf_idx = LEG_TORSO_URDF_IDS[i];
            int motor_idx = env->robot->data.joint_ids_map[urdf_idx];
            lowcmd->msg_.motor_cmd()[motor_idx].q() = action[i];
    }
    } else {
        // 27-action policy (legacy / push_v4): direct articulation-order mapping.
        // action[i] corresponds to articulation index i, joint_ids_map[i] is its motor.
        size_t n = std::min(action.size(), env->robot->data.joint_ids_map.size());
        for (size_t i = 0; i < n; i++) {
            lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
        }
    }

    // For Option II (13-action) policies: write arm joint targets from the
    // publisher. This overrides whatever the action loop wrote to arm motors
    // (which would be garbage / out-of-range for 13-action policies anyway).
    if (action.size() < 27) {
        auto& pub = h1_2::ArmPosePublisher::instance();
        // dt is measured from the wall clock inside compute_arm_targets (the
        // arg is ignored), so the slew/wobble are correct at this 1 kHz loop.
        auto arm_targets = pub.compute_arm_targets(env->step_dt);
        const bool gain_override = pub.arm_gain_override();
        for (size_t i = 0; i < 14; i++) {
            auto& cmd = lowcmd->msg_.motor_cmd()[ARM_SDK_MOTOR_IDS[i]];
            cmd.q() = arm_targets[i];
            if (gain_override) {
                // Per-joint arm gains, re-asserted every step (enter() writes
                // the deploy.yaml gains on every entry into this state).
                cmd.kp() = pub.arm_kp(i);
                cmd.kd() = pub.arm_kd(i);
            }
        }
    }
}
