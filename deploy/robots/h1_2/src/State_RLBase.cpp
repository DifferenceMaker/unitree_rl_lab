#include <iostream>
#include <cmath>
#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

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
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();

    // DEBUG: log action + obs values once per second
    static int log_counter = 0;
    if (log_counter++ % 50 == 0) {
        // Print first 3 action values
        std::cout << "[RL] action[0..2]=" << action[0] << "," << action[1] << "," << action[2];

        // Try to peek into the observation manager
        try {
            auto obs_map = env->observation_manager->compute();
            for (const auto& kv : obs_map) {
                const auto& flat = kv.second;
                std::cout << " obs[" << kv.first << "].size=" << flat.size()
                          << " [0..3]=" << flat[0] << "," << flat[1] << "," << flat[2];
                // Count NaNs in obs
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

    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
