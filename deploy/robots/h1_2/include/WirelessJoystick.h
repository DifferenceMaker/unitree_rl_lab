#pragma once
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

namespace h1_2 {

// Walk sim2sim (2026-08-12): a UnitreeJoystick fed from the rt/wirelesscontroller
// DDS topic instead of the lowstate wireless_remote bytes.
//
// WHY: architecture B is joystickless — the MuJoCo sim publishes lowstate with
// zeroed remote bytes, so lowstate->joystick reads all-zero. The walk policy's
// velocity_commands observation needs live stick values. A keyboard teleop tab
// (tools/walk_teleop.py) publishes plain float lx/ly/rx/ry on
// rt/wirelesscontroller — the same topic the REAL remote publishes natively on
// hardware — so this adapter is sim2real-faithful by construction: on the
// robot, the physical remote drives the identical code path.
// Same trust level as the gamepad (operator-initiated), like FsmCmdSubscriber.
class WirelessJoystick : public unitree::common::UnitreeJoystick {
public:
    WirelessJoystick()
        : sub_(std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>>(
              "rt/wirelesscontroller",
              [this](const void* msg) {
                  const auto& m = *reinterpret_cast<const unitree_go::msg::dds_::WirelessController_*>(msg);
                  lx(m.lx()); ly(m.ly()); rx(m.rx()); ry(m.ry());
                  // xKeySwitchUnion bit layout (same as the remote encodes):
                  const uint16_t k = m.keys();
                  RB(k & 0x0001); LB(k & 0x0002); start(k & 0x0004); back(k & 0x0008);
                  A(k & 0x0100); B(k & 0x0200); X(k & 0x0400); Y(k & 0x0800);
                  up(k & 0x1000); right(k & 0x2000); down(k & 0x4000); left(k & 0x8000);
              })) {}

private:
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> sub_;
};

}  // namespace h1_2
