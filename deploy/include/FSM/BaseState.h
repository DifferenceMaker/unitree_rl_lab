// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once
#include <atomic>
#include <vector>
#include <utility>

#include <boost/bimap.hpp>
#include <string>
#include <any>
#include <utility>

inline boost::bimap<int, std::string> FSMStringMap;

// Joystickless FSM override (sim2sim harness): stdin `fsm <name>` sets this
// to a state id; CtrlFSM consumes it on the next 1 ms tick and performs the
// normal exit()/enter() transition. 0 = no request.
inline std::atomic<int> FSMRequest{0};

// Number-key FSM map (sim2sim HUD/keyboard): '0'=Passive, '1'=FixStand,
// '2'.. = remaining states in ascending id order. Shared by the controller
// (rt/fsm_cmd handler) and the HUD string so both sides agree by construction.
inline std::vector<std::pair<char, int>> fsm_key_map()
{
    std::vector<std::pair<char, int>> map;
    int passive = FSMStringMap.right.count("Passive") ? FSMStringMap.right.at("Passive") : 0;
    int fixstand = FSMStringMap.right.count("FixStand") ? FSMStringMap.right.at("FixStand") : 0;
    if (passive)  map.push_back({'0', passive});
    if (fixstand) map.push_back({'1', fixstand});
    char key = '2';
    for (auto& kv : FSMStringMap.left) {           // ascending id
        if (kv.first == passive || kv.first == fixstand) continue;
        if (key > '9') break;                       // full digit range: FSM input
        // comes from the NUMPAD (sim, bd5645a) / stdin — the elastic-band keys
        // are MAIN-ROW 7/8/9 only, so the old '9'-reserved cap is obsolete.
        map.push_back({key++, kv.first});
    }
    return map;
}

inline std::string fsm_keys_string()
{
    std::string s;
    auto keymap = fsm_key_map();
    for (auto& [key, id] : keymap) {
        if (!s.empty()) s += "  ";
        s += key; s += "="; s += FSMStringMap.left.at(id);
    }
    // States beyond the key range: published with '-' (stdin `fsm <name>` only)
    // so the HUD always shows the COMPLETE state list.
    for (auto& kv : FSMStringMap.left) {
        bool keyed = false;
        for (auto& [k, id] : keymap) if (id == kv.first) { keyed = true; break; }
        if (!keyed) { if (!s.empty()) s += "  "; s += "-="; s += kv.second; }
    }
    return s;
}

inline int fsm_state_for_key(char c)
{
    for (auto& [key, id] : fsm_key_map())
        if (key == c) return id;
    return 0;
}

class BaseState
{
public:
    BaseState(int state, std::string state_string) : state_(state) 
    {
        FSMStringMap.insert({state, state_string});
    }

    virtual void enter() {}

    virtual void pre_run() {}
    virtual void run() {}
    virtual void post_run() {}

    virtual void exit() {}

    std::string getStateString() { return FSMStringMap.left.at(state_); }
    int getState() {return state_; }
    bool isState(int state) { return state_ == state; }
    std::vector<std::pair<std::function<bool()>, int>> registered_checks;
private:
    int state_;
};

using FsmFactory = std::function<std::shared_ptr<BaseState>(int, std::string)>;
using FsmMap     = std::unordered_map<std::string, FsmFactory>;

inline FsmMap& getFsmMap() {
    static FsmMap fsmMap;
    return fsmMap;
}

#define REGISTER_FSM(Derived) \
    inline std::shared_ptr<BaseState> __factory_##Derived(int s, std::string ss) {      \
        return std::make_shared<Derived>(s, ss);                                        \
    }                                                                                   \
    inline struct __registrar_##Derived {                                               \
        __registrar_##Derived() {                                                       \
            getFsmMap()[#Derived] = __factory_##Derived;                                \
        }                                                                               \
    } __registrar_instance_##Derived;                                                   \
    
