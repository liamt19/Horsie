
#include "dbg_hit.h"

#include <array>
#include <atomic>
#include <iostream>

constexpr int MaxDebugSlots = 32;

//  See https://github.com/official-stockfish/Stockfish/blob/c085670b8474dd2137446ff278f6b73f4374cc68/src/misc.cpp#L283
namespace {

    template<size_t N>
    struct DebugInfo {
        std::atomic<i64> data[N] = { 0 };

        [[nodiscard]] constexpr std::atomic<i64>& operator[](size_t index) {
            return data[index];
        }
    };

    std::array<DebugInfo<2>, MaxDebugSlots>  hit;
}

namespace Horsie {
    void DbgHit(bool cond, i32 slot) {
        ++hit.at(slot)[0];
        if (cond)
            ++hit.at(slot)[1];
    }

    void DbgPrint() {
        i64 n;
        auto E = [&n](i64 x) { return double(x) / n; };

        for (int i = 0; i < MaxDebugSlots; ++i)
            if ((n = hit[i][0]))
                std::cout << "Hit #" << i << ": Total " << n << " Hits " << hit[i][1] << " Hit Rate (%) " << 100.0 * E(hit[i][1]) << std::endl;
    }
}
