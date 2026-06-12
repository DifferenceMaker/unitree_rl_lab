// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include "isaaclab/utils/utils.h"

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        _ema_prev.clear();
        robot->update();
        action_manager->reset();
        observation_manager->reset();
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        auto obs = observation_manager->compute();
        auto action = alg->act(obs);

        // Optional EMA action filter (sim2sim filter tuning ONLY — keep
        // action_ema_alpha at 0.0 for any training-comparison run, since a
        // filtered action path is not what the policy was trained against):
        //   a_smooth = alpha * a_prev + (1 - alpha) * a_new
        // Applied to raw policy outputs BEFORE scale/offset/clip. The
        // last_action observation keeps the UNfiltered action (raw/executed
        // split in ActionManager) so the obs contract matches training.
        if (action_ema_alpha > 0.0f) {
            if (_ema_prev.size() != action.size()) _ema_prev = action;
            std::vector<float> smoothed(action.size());
            for (size_t i = 0; i < action.size(); ++i) {
                smoothed[i] = action_ema_alpha * _ema_prev[i]
                            + (1.0f - action_ema_alpha) * action[i];
            }
            _ema_prev = smoothed;
            action_manager->process_action(action, smoothed);
        } else {
            action_manager->process_action(action);
        }
    }

    // 0.0 = filter off (default). Set from the robot controller's config.yaml
    // (NOT from deploy.yaml — per-milestone params stay untouched).
    float action_ema_alpha = 0.0f;

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;

private:
    // EMA filter state (previous smoothed action); cleared on reset
    std::vector<float> _ema_prev;
};

};