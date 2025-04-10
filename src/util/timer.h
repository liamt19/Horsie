#pragma once

#include <chrono>

namespace Horsie {

    class Timepoint {
    private:
        using Clock = std::chrono::steady_clock;
        using Milliseconds = std::chrono::milliseconds;
        using MillisType = i64;

        Clock::time_point tp{}; 
    
    public:

        Timepoint() = default;
        explicit Timepoint(Clock::time_point t) : tp(t) {}

        static Timepoint Now() {
            return Timepoint(Clock::now());
        }

        static MillisType TimeSince(const Timepoint& start) {
            return std::chrono::duration_cast<Milliseconds>(Now().tp - start.tp).count();
        }

        static MillisType TimeBetween(const Timepoint& start, const Timepoint& end) {
            return std::chrono::duration_cast<Milliseconds>(end.tp - start.tp).count();
        }

        static std::pair<u64, u64> UnpackSecondsMillis(const Timepoint& start) {
            const auto duration = TimeSince(start);
            return { duration / 1000,  duration % 1000 };
        }

        static std::pair<u64, u64> UnpackSecondsMillis(MillisType duration) {
            return { duration / 1000, duration % 1000 };
        }

        template<class T>
        static u64 NPS(T amount, MillisType duration) {
            const double n = amount / (duration / static_cast<double>(1000));
            return static_cast<u64>(n);
        }
    };
}
