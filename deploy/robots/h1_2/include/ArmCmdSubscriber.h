#pragma once

#include <array>
#include <memory>
#include <sstream>
#include <string>

#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/ros2/String_.hpp>

#include "ArmPosePublisher.h"

namespace h1_2 {

// Subscribes to rt/arm_pose_cmd and drives ArmPosePublisher into External mode.
// This is the decoupled arm-command interface: an external source (MuJoCo GUI,
// a script, teleop, or — on the real robot — a manipulation/task module) publishes
// the desired 14-dim arm pose and the controller slews to it, feeding obs+motors
// exactly as it always has. The balance policy is unchanged — it reads the arm
// command from its observation regardless of where the command originates.
//
// Payload JSON: {"pose":[14 floats],"transition_s":<optional float>}
class ArmCmdSubscriber {
public:
    ArmCmdSubscriber()
        : sub_(std::make_shared<unitree::robot::ChannelSubscriber<std_msgs::msg::dds_::String_>>(
              "rt/arm_pose_cmd", [](const void *msg) {
                  handle(reinterpret_cast<const std_msgs::msg::dds_::String_ *>(msg)->data());
              })) {
        sub_->InitChannel();
    }

private:
    static void handle(const std::string &js) {
        std::array<float, 14> pose;
        if (!parse_pose(js, pose)) {
            return;  // malformed / missing pose — ignore (don't disturb current arm state)
        }
        float trans;
        if (parse_scalar(js, "transition_s", trans) && trans >= 0.0f) {
            ArmPosePublisher::instance().set_transition_s(trans);
        }
        ArmPosePublisher::instance().set_external_pose(pose);
    }

    // Parse "pose":[v0,v1,...,v13] — requires exactly 14 numbers.
    static bool parse_pose(const std::string &js, std::array<float, 14> &out) {
        auto p = js.find("\"pose\"");
        if (p == std::string::npos) return false;
        p = js.find('[', p);
        if (p == std::string::npos) return false;
        auto e = js.find(']', p);
        if (e == std::string::npos) return false;
        std::string body = js.substr(p + 1, e - (p + 1));
        for (char &c : body) {
            if (c == ',') c = ' ';
        }
        std::istringstream iss(body);
        size_t n = 0;
        float v;
        while (n < 14 && (iss >> v)) out[n++] = v;
        return n == 14;
    }

    // Parse "key":<number> (scalar). Returns false if absent / non-numeric.
    static bool parse_scalar(const std::string &js, const std::string &key, float &out) {
        const std::string k = "\"" + key + "\"";
        auto p = js.find(k);
        if (p == std::string::npos) return false;
        p = js.find(':', p + k.size());
        if (p == std::string::npos) return false;
        std::istringstream iss(js.substr(p + 1));
        return static_cast<bool>(iss >> out);
    }

    std::shared_ptr<unitree::robot::ChannelSubscriber<std_msgs::msg::dds_::String_>> sub_;
};

}  // namespace h1_2
