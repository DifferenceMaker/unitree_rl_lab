// Walk sim2sim state (2026-08-12) — the GENERIC whole-body RL state, taken
// from the upstream g1_29dof State_RLBase shape. Deliberately NOT the h1_2
// State_RLBase: that one is hardwired Option-II (13-action legs+torso, arms
// via ArmPosePublisher + dpad presets), while walk policies command all 27
// joints through the deploy.yaml action contract.
//
// First-iteration simplifications, on purpose:
//   * no engage blend — walk is entered from FixStand under the sim's elastic
//     band; the arm snap to the walk default pose is acceptable in sim. Add
//     the RLBase blend before any hardware use (2026-07-08 snap incident).
//   * no arm publisher, no latency window, no dpad handling.
#pragma once

#include <chrono>
#include <ctime>
#include <iostream>
#include "FSM/FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_Walk : public FSMState
{
public:
    State_Walk(int state_mode, std::string state_string);

    void enter()
    {
        auto now = std::chrono::system_clock::now();
        auto now_t = std::chrono::system_clock::to_time_t(now);
        std::cout << "============================================================" << std::endl;
        std::cout << "[FSM] ENTER " << getStateString() << " (whole-body walk) — policy:" << std::endl;
        std::cout << "[FSM]   " << param::config["FSM"][getStateString()]["policy_dir"].as<std::string>() << std::endl;
        std::cout << "[FSM] State ID: " << getState() << "  Entered at: " << std::ctime(&now_t);
        std::cout << "[FSM] velocity source: rt/wirelesscontroller (tools/walk_teleop.py or real remote)" << std::endl;
        std::cout << "============================================================" << std::endl;

        // gains from the policy's own deploy.yaml (walk: 200/300 legs, 40 arms)
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);
            auto sleepTill = clock::now() + dt;
            env->reset();
            while (policy_thread_running)
            {
                env->step();
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();

    void exit()
    {
        std::cout << "============================================================" << std::endl;
        std::cout << "[FSM] EXIT " << getStateString() << std::endl;
        std::cout << "============================================================" << std::endl;
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_Walk)
