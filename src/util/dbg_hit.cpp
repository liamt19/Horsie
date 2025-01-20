
#include "dbg_hit.h"

#include <array>
#include <atomic>
#include <iostream>

constexpr int MaxDebugSlots = 32;

//  See https://github.com/official-stockfish/Stockfish/blob/c085670b8474dd2137446ff278f6b73f4374cc68/src/misc.cpp#L283
namespace {

    template<size_t N>
    struct DebugInfo {
        std::atomic<int64_t> data[N] = { 0 };

        [[nodiscard]] constexpr std::atomic<int64_t>& operator[](size_t index) {
            assert(index < N);
            return data[index];
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
    std::array<DebugInfo<3>, MaxDebugSlots>  stdev;
    std::array<DebugInfo<6>, MaxDebugSlots>  correl;
    std::array<DebugExtremes, MaxDebugSlots> extremes;

}  // namespace

namespace Horsie {
    void dbg_hit_on(bool cond, int slot) {
        ++hit.at(slot)[0];
        if (cond)
            ++hit.at(slot)[1];
    }

    void dbg_mean_of(int64_t value, int slot) {
        ++mean.at(slot)[0];
        mean.at(slot)[1] += value;
    }

    void dbg_stdev_of(int64_t value, int slot) {
        ++stdev.at(slot)[0];
        stdev.at(slot)[1] += value;
        stdev.at(slot)[2] += value * value;
    }

    void dbg_extremes_of(int64_t value, int slot) {
        ++extremes.at(slot)[0];

        int64_t current_max = extremes.at(slot)[1].load();
        while (current_max < value && !extremes.at(slot)[1].compare_exchange_weak(current_max, value)) {
        }

        int64_t current_min = extremes.at(slot)[2].load();
        while (current_min > value && !extremes.at(slot)[2].compare_exchange_weak(current_min, value)) {
        }
    }

    void dbg_correl_of(int64_t value1, int64_t value2, int slot) {
        ++correl.at(slot)[0];
        correl.at(slot)[1] += value1;
        correl.at(slot)[2] += value1 * value1;
        correl.at(slot)[3] += value2;
        correl.at(slot)[4] += value2 * value2;
        correl.at(slot)[5] += value1 * value2;
    }

    void dbg_print() {
        int64_t n;
        auto    E = [&n](int64_t x) { return double(x) / n; };
        auto    sqr = [](double x) { return x * x; };

        for (int i = 0; i < MaxDebugSlots; ++i)
            if ((n = hit[i][0]))
                std::cerr << "Hit #" << i << ": Total " << n << " Hits " << hit[i][1]
                << " Hit Rate (%) " << 100.0 * E(hit[i][1]) << std::endl;

        for (int i = 0; i < MaxDebugSlots; ++i)
            if ((n = mean[i][0])) {
                std::cerr << "Mean #" << i << ": Total " << n << " Mean " << E(mean[i][1]) << std::endl;
            }

        for (int i = 0; i < MaxDebugSlots; ++i)
            if ((n = stdev[i][0])) {
                double r = sqrt(E(stdev[i][2]) - sqr(E(stdev[i][1])));
                std::cerr << "Stdev #" << i << ": Total " << n << " Stdev " << r << std::endl;
            }

        for (int i = 0; i < MaxDebugSlots; ++i)
            if ((n = extremes[i][0])) {
                std::cerr << "Extremity #" << i << ": Total " << n << " Min " << extremes[i][2]
                    << " Max " << extremes[i][1] << std::endl;
            }

        for (int i = 0; i < MaxDebugSlots; ++i)
            if ((n = correl[i][0])) {
                double r = (E(correl[i][5]) - E(correl[i][1]) * E(correl[i][3]))
                    / (sqrt(E(correl[i][2]) - sqr(E(correl[i][1])))
                       * sqrt(E(correl[i][4]) - sqr(E(correl[i][3]))));
                std::cerr << "Correl. #" << i << ": Total " << n << " Coefficient " << r << std::endl;
            }
    }
}