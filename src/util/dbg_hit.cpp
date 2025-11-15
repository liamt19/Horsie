
#include "dbg_hit.h"

#include <array>
#include <atomic>
#include <iostream>

constexpr size_t MaxDebugSlots = 64;

//  See https://github.com/official-stockfish/Stockfish/blob/c085670b8474dd2137446ff278f6b73f4374cc68/src/misc.cpp#L283
namespace {

    template<size_t N>
    struct DebugInfo {
        std::array<std::atomic<int64_t>, N> data = { 0 };

        constexpr std::atomic<int64_t>& operator[](size_t index) {
            return data[index];
        }

        constexpr DebugInfo& operator=(const DebugInfo& other) {
            for (size_t i = 0; i < N; i++)
                data[i].store(other.data[i].load());
            return *this;
        }
    };

    struct DebugExtremes : public DebugInfo<3> {
        DebugExtremes() {
            data[1] = std::numeric_limits<int64_t>::min();
            data[2] = std::numeric_limits<int64_t>::max();
        }
    };

    std::array<DebugInfo<2>, MaxDebugSlots>  hit;
    std::array<DebugInfo<2>, MaxDebugSlots>  mean;
    std::array<DebugExtremes, MaxDebugSlots> extremes;

}

namespace Horsie {
    void DbgHit(bool cond, i32 slot) {
        ++hit.at(slot)[0];
        if (cond)
            ++hit.at(slot)[1];
    }

    void DbgMean(i64 value, i32 slot) {
        ++mean.at(slot)[0];
        mean.at(slot)[1] += value;
    }

    void DbgMinMax(i64 value, i32 slot) {
        ++extremes.at(slot)[0];
        i64 current_max = extremes.at(slot)[1].load();
        while (current_max < value && !extremes.at(slot)[1].compare_exchange_weak(current_max, value)) {}
        i64 current_min = extremes.at(slot)[2].load();
        while (current_min > value && !extremes.at(slot)[2].compare_exchange_weak(current_min, value)) {}
    }

    void DbgPrint() {
        int64_t n;
        auto    E = [&n](int64_t x) { return double(x) / n; };

        for (int i = 0; i < MaxDebugSlots; ++i)
            if ((n = hit[i][0]))
                std::cout << "Hit #" << i << ": Total " << n << " Hits " << hit[i][1] << " Hit Rate (%) " << 100.0 * E(hit[i][1]) << std::endl;

        for (int i = 0; i < MaxDebugSlots; ++i)
            if ((n = mean[i][0])) {
                std::cout << "Mean #" << i << ": Total " << n << " Mean " << E(mean[i][1]) << std::endl;
            }

        for (int i = 0; i < MaxDebugSlots; ++i)
            if ((n = extremes[i][0])) {
                std::cout << "Extremity #" << i << ": Total " << n << " Min " << extremes[i][2] << " Max " << extremes[i][1] << std::endl;
            }
    }
}
