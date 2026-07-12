#pragma once

#include <array>
#include <memory>
#include <sstream>
#include <string>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/ros2/String_.hpp>
#include "FSM/BaseState.h"

#include "ArmPosePublisher.h"

namespace h1_2 {

// Sim2sim / debug HUD publisher. Emits the active policy name + the resolved arm
// gains as a small JSON string on the `rt/policy_status` topic, which the MuJoCo
// sim subscribes to and overlays on-screen. Publishes immediately when the policy
// changes and on a low-rate heartbeat otherwise (so a late-joining sim still gets
// the current state). DEBUG-ONLY: completely separate from the lowcmd control path.
//
// JSON shape:
//   {"policy":"Balance_perjoint","arm_transition_s":4.0,"arm_gain_override":true,
//    "arm_kp":[...14...],"arm_kd":[...14...]}
class PolicyStatusPublisher {
public:
    static PolicyStatusPublisher& instance() {
        static PolicyStatusPublisher inst;
        return inst;
    }

    // Call every control tick with the active FSM state name. Throttles internally.
    void publish(const std::string& policy, bool force = false) {
        const bool changed = (policy != last_policy_);
        if (!force && !changed && (++tick_ % kHeartbeatTicks) != 0) {
            return;
        }
        last_policy_ = policy;

        auto& arm = ArmPosePublisher::instance();
        std::ostringstream js;
        js << "{\"policy\":\"" << policy << "\","
           << "\"arm_transition_s\":" << arm.transition_s() << ","
           << "\"arm_gain_override\":" << (arm.arm_gain_override() ? "true" : "false") << ","
           << "\"arm_kp\":[";
        for (size_t i = 0; i < 14; ++i) js << (i ? "," : "") << arm.arm_kp(i);
        js << "],\"arm_kd\":[";
        for (size_t i = 0; i < 14; ++i) js << (i ? "," : "") << arm.arm_kd(i);
        js << "],\"fsm_keys\":\"" << fsm_keys_string() << "\"}";

        msg_.data(js.str());
        pub_->Write(msg_, 0);
    }

private:
    PolicyStatusPublisher()
        : pub_(std::make_unique<unitree::robot::ChannelPublisher<std_msgs::msg::dds_::String_>>(
              "rt/policy_status")) {
        pub_->InitChannel();
    }
    PolicyStatusPublisher(const PolicyStatusPublisher&) = delete;
    PolicyStatusPublisher& operator=(const PolicyStatusPublisher&) = delete;

    // Heartbeat period in run() ticks (publish at least this often even when the
    // policy is unchanged, so a sim that connects late picks up the current state).
    static constexpr int kHeartbeatTicks = 10;

    std::unique_ptr<unitree::robot::ChannelPublisher<std_msgs::msg::dds_::String_>> pub_;
    std_msgs::msg::dds_::String_ msg_;
    std::string last_policy_;
    int tick_ = 0;
};

}  // namespace h1_2
