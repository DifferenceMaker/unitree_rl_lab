#include <iostream>
#include <cmath>
#include <algorithm>
#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "ArmPosePublisher.h"

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

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );

    // Phase 5 v3 is Option II: 13-action policy controlling only legs+torso.
    // Arms driven externally by ArmPosePublisher. Initialize to Training mode
    // (matches training distribution: held pose ±1.5/±1.0/±1.5, wobble 0.15).
    h1_2::ArmPosePublisher::instance().set_mode(h1_2::ArmPosePublisher::Mode::Idle);

    std::cout << "[FSM] State_RLBase " << state_string << " constructed." << std::endl;
    std::cout << "[FSM]   Arm mode default: IDLE (no motion). DPad to change:" << std::endl;
    std::cout << "[FSM]   Up=Idle | Right=Mild | Down=Training" << std::endl;
}

void State_RLBase::run()
{
    // ===== Arm mode switching via DPad (rising-edge triggered) =====
    // Up=Idle, Right=Mild, Down=Training, Left=reserved
    // Operates independent of FSM state — works in any BalancePush variant.
    using ArmMode = h1_2::ArmPosePublisher::Mode;
    auto& arm_pub = h1_2::ArmPosePublisher::instance();

    if (lowstate->joystick.up.on_pressed) {
        arm_pub.set_mode(ArmMode::Idle);
        std::cout << "[ARM_MODE] -> IDLE (no motion)" << std::endl;
    }
    if (lowstate->joystick.right.on_pressed) {
        arm_pub.set_mode(ArmMode::Mild);
        std::cout << "[ARM_MODE] -> MILD (gentle disturbance)" << std::endl;
    }
    if (lowstate->joystick.down.on_pressed) {
        arm_pub.set_mode(ArmMode::Training);
        std::cout << "[ARM_MODE] -> TRAINING (full disturbance)" << std::endl;
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
        auto arm_targets = h1_2::ArmPosePublisher::instance().compute_arm_targets(env->step_dt);
        for (size_t i = 0; i < 14; i++) {
            lowcmd->msg_.motor_cmd()[ARM_SDK_MOTOR_IDS[i]].q() = arm_targets[i];
        }
    }
}
