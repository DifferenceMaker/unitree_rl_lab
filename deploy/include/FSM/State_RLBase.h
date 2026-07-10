// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <chrono>
#include <ctime>
#include <iostream>
#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "ArmPosePublisher.h"
#include "LatencyStats.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
         // Log state entry — fires on every transition INTO this state
        auto now = std::chrono::system_clock::now();
        auto now_t = std::chrono::system_clock::to_time_t(now);
        std::cout << "============================================================" << std::endl;
        std::cout << "[FSM] ENTER " << getStateString() << " — policy:" << std::endl;
        std::cout << "[FSM]   " << param::config["FSM"][getStateString()]["policy_dir"].as<std::string>() << std::endl;
        std::cout << "[FSM] State ID: " << getState() << "  Entered at: " << std::ctime(&now_t);
        std::cout << "============================================================" << std::endl;

        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

        // Engage blend (2026-07-08 hip-yaw snap incident): on entry, arm the
        // measured-pose capture; run() blends the written leg+torso targets
        // from the measured joints to the policy targets over engage_blend_s.
        // Prevents the FixStand->policy-stance snap that tripped the motor
        // protection on hardware. Arms already ramp via arm_transition_s.
        engage_pending = true;
        engage_t0 = std::chrono::steady_clock::now();

        // [dds_cpp latency] stats window active only inside RLBase states.
        latency_seq_consumed = latency::Stats::instance().action_seq.load(std::memory_order_relaxed);
        latency::Stats::instance().enabled.store(true, std::memory_order_relaxed);

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();
 
            while (policy_thread_running)
            {
                env->step();

                // Sleep
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
        
        latency::Stats::instance().enabled.store(false, std::memory_order_relaxed);

        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;

    // Engage blend state (see enter()). engage_blend_s is read from the
    // state's config in the constructor (default 1.5 s; <=0 disables).
    float engage_blend_s = 1.5f;
    bool engage_pending = false;
    uint64_t latency_seq_consumed = 0;   // [dds_cpp latency] run()-thread only
    std::array<float, 13> engage_q0{};
    std::chrono::steady_clock::time_point engage_t0;
};

REGISTER_FSM(State_RLBase)
