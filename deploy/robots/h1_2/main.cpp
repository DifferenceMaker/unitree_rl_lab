#include <array>
#include <sstream>
#include <string>

#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "ArmPosePublisher.h"
#include "ArmCmdSubscriber.h"
#include "LatencyStats.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    // StampedLowState: identical behavior + steady-clock arrival stamp for
    // the [dds_cpp] latency stats (LatencyStats.h).
    FSMState::lowstate = std::make_shared<latency::StampedLowState>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     H1-2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 6;
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    // Decoupled arm-command interface: listen on rt/arm_pose_cmd and drive the
    // ArmPosePublisher into External mode (MuJoCo GUI / script / teleop sends poses).
    auto arm_cmd_sub = std::make_unique<h1_2::ArmCmdSubscriber>();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";

    // Harness stdin commands (sim2sim measurement tool):
    //   arm <14 floats>   set a manual held arm pose (radians, URDF arm order:
    //                     shPitch L R, shRoll L R, shYaw L R, elbPitch L R,
    //                     elbRoll L R, wrPitch L R, wrYaw L R). Switches the
    //                     arm source to Manual; targets slew there (no jump).
    //   arm default       manual pose = default arm pose
    std::string line;
    while (std::getline(std::cin, line))
    {
        std::istringstream iss(line);
        std::string cmd;
        if (!(iss >> cmd) || cmd.empty()) continue;

        if (cmd == "arm") {
            std::string first;
            if (!(iss >> first)) {
                std::cout << "[CMD] usage: arm <14 floats> | arm default" << std::endl;
                continue;
            }
            auto& arm_pub = h1_2::ArmPosePublisher::instance();
            if (first == "default") {
                arm_pub.set_manual_pose(h1_2::ArmPosePublisher::default_arm_pos());
                std::cout << "[CMD] arm -> MANUAL (default pose)" << std::endl;
                continue;
            }
            std::array<float, 14> pose;
            try { pose[0] = std::stof(first); }
            catch (...) { std::cout << "[CMD] arm: bad value '" << first << "'" << std::endl; continue; }
            size_t n = 1;
            float v;
            while (n < 14 && (iss >> v)) pose[n++] = v;
            if (n != 14) {
                std::cout << "[CMD] arm: need 14 values, got " << n << std::endl;
                continue;
            }
            arm_pub.set_manual_pose(pose);
            std::cout << "[CMD] arm -> MANUAL pose set (slewing there)" << std::endl;
        } else {
            std::cout << "[CMD] unknown: '" << cmd << "' (commands: arm)" << std::endl;
        }
    }

    // stdin closed (piped/backgrounded run) — keep the controller alive.
    while (true)
    {
        sleep(1);
    }

    return 0;
}

