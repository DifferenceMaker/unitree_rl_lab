// Latency instrumentation for ROS-vs-dds_cpp measured-vs-measured comparison.
// Mirrors Aspired_Robot_Project (main_arch_b_test) BridgeModule/MovementModule
// probes:
//   obs_age    = rt/lowstate DDS arrival -> the moment the observation builder
//                reads motor_state/IMU (BaseArticulation::update, policy thread)
//   action_age = policy output written (ManagerBasedRLEnv::step, ~50 Hz)
//                -> the 1 kHz FSM run() consuming that output into lowcmd
//                (each policy output counted ONCE; 1 kHz re-writes of a held
//                target are not new actions)
//   loop mean  = obs mean + action mean
// 1 Hz print, format matched to the ROS stack's "[Bridge set] ..." line:
//   [dds_cpp] action_age ms: min=.. mean=.. max=.. (n=..) | obs_age ms: min=..
//   mean=.. max=.. | loop mean=..
//
// Rules honored: steady_clock ONLY (no DDS/source/wall timestamps); producer
// stamps are lock-free atomics (the lowstate stamp is set inside the SDK
// callback's existing mutex via the post_communication hook — no new locks on
// any hot path; the stats mutex is touched at ~50 Hz by samplers and 1 Hz by
// the printer, never per-1kHz-tick); print-only — zero behavioral changes;
// sampling is gated to RLBase (Balance) states via `enabled`.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>

#include "Types.h"

namespace latency {

inline int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

class Stats
{
public:
    static Stats& instance()
    {
        static Stats s;
        return s;
    }

    // ---- producer stamps (lock-free) ----
    std::atomic<int64_t> lowstate_arrival_ns{0};   // set in DDS callback hook
    std::atomic<int64_t> action_stamp_ns{0};       // set when policy output updated
    std::atomic<uint64_t> action_seq{0};           // bumped per policy output
    std::atomic<bool> enabled{false};              // true only inside RLBase states

    // Policy thread, ~50 Hz: called when the obs builder reads the cached lowstate.
    void sample_obs_age()
    {
        if (!enabled.load(std::memory_order_relaxed)) return;
        int64_t arr = lowstate_arrival_ns.load(std::memory_order_relaxed);
        if (arr == 0) return;
        double ms = (now_ns() - arr) / 1e6;
        std::lock_guard<std::mutex> lock(m_);
        obs_.add(ms);
    }

    // FSM run() thread, on FIRST consumption of a new policy output (~50 Hz).
    void sample_action_age()
    {
        int64_t st = action_stamp_ns.load(std::memory_order_relaxed);
        if (st == 0) return;
        double ms = (now_ns() - st) / 1e6;
        std::lock_guard<std::mutex> lock(m_);
        act_.add(ms);
    }

    // FSM run() thread (single caller). Prints + resets the 1 s window.
    void maybe_print()
    {
        if (!enabled.load(std::memory_order_relaxed)) return;
        int64_t now = now_ns();
        if (now - last_print_ns_ < 1'000'000'000LL) return;
        last_print_ns_ = now;

        Acc obs, act;
        {
            std::lock_guard<std::mutex> lock(m_);
            obs = obs_; act = act_;
            obs_.reset(); act_.reset();
        }
        if (act.n == 0) return;   // nothing consumed this window (mirrors reference)
        char line[256];
        int off = std::snprintf(line, sizeof(line),
            "[dds_cpp] action_age ms: min=%.1f mean=%.1f max=%.1f (n=%d)",
            act.mn, act.mean(), act.mx, act.n);
        if (obs.n > 0) {
            std::snprintf(line + off, sizeof(line) - off,
                " | obs_age ms: min=%.1f mean=%.1f max=%.1f | loop mean=%.1f",
                obs.mn, obs.mean(), obs.mx, obs.mean() + act.mean());
        }
        std::printf("%s\n", line);
        std::fflush(stdout);
    }

private:
    struct Acc
    {
        double mn = std::numeric_limits<double>::max();
        double mx = 0.0, sum = 0.0;
        int n = 0;
        void add(double v) { if (v < mn) mn = v; if (v > mx) mx = v; sum += v; ++n; }
        double mean() const { return n ? sum / n : 0.0; }
        void reset() { *this = Acc{}; }
    };

    std::mutex m_;            // guards the two accumulators (50 Hz + 1 Hz touch rate)
    Acc obs_, act_;
    int64_t last_print_ns_ = 0;   // run()-thread-only
};

// rt/lowstate subscription that stamps message ARRIVAL on the steady clock.
// post_communication() runs inside SubscriptionBase's DDS callback, under its
// existing mutex_ — the atomic store adds no locking.
class StampedLowState : public LowState_t
{
public:
    using LowState_t::LowState_t;

protected:
    void post_communication() override
    {
        Stats::instance().lowstate_arrival_ns.store(now_ns(), std::memory_order_relaxed);
    }
};

} // namespace latency
