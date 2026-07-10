#pragma once

#include <string>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/ros2/String_.hpp>
#include "FSM/BaseState.h"

namespace h1_2 {

// Sim2sim keyboard FSM control: the MuJoCo sim publishes a key digit (or a
// state name) on rt/fsm_cmd when the operator presses 0-8 in the sim window;
// this maps it through fsm_key_map() / FSMStringMap into FSMRequest — the
// same joystickless path as the stdin `fsm` command (normal exit()/enter()).
// DEBUG/sim tool: same trust level as the gamepad (operator-initiated).
class FsmCmdSubscriber {
public:
    FsmCmdSubscriber()
        : sub_(std::make_shared<unitree::robot::ChannelSubscriber<std_msgs::msg::dds_::String_>>(
              "rt/fsm_cmd",
              [](const void* msg) {
                  const auto& s = reinterpret_cast<const std_msgs::msg::dds_::String_*>(msg)->data();
                  if (s.empty()) return;
                  int id = 0;
                  if (s.size() == 1) {
                      id = fsm_state_for_key(s[0]);
                  } else if (FSMStringMap.right.count(s)) {
                      id = FSMStringMap.right.at(s);
                  }
                  if (id != 0) {
                      FSMRequest.store(id);
                      std::cout << "[CMD] rt/fsm_cmd -> requested "
                                << FSMStringMap.left.at(id) << std::endl;
                  } else {
                      std::cout << "[CMD] rt/fsm_cmd: unknown '" << s << "'" << std::endl;
                  }
              }))
    {
        sub_->InitChannel();
    }

private:
    std::shared_ptr<unitree::robot::ChannelSubscriber<std_msgs::msg::dds_::String_>> sub_;
};

} // namespace h1_2
