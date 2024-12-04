#pragma once

#ifndef HISTORY_H
#define HISTORY_H

#include <algorithm>
#include <cstring>

#include "util/NDArray.h"


namespace Horsie {

    constexpr i32 HistoryClamp = 16384;
    constexpr i32 LowPlyCount = 4;
    constexpr i32 LowPlyClamp = 8192;


    template<typename T, i32 ClampVal>
    class StatsEntry {
        T entry;

        public:
            void operator=(const T& v) { entry = v; }
            T* operator&() { return &entry; }
            T* operator->() { return &entry; }
            operator const T& () const { return entry; }

            void operator<<(i32 bonus) {
                static_assert(ClampVal <= std::numeric_limits<T>::max(), "D overflows T");
                entry += (bonus - (entry * std::abs(bonus) / ClampVal));
                assert(std::abs(entry) <= ClampVal);
            }
    };

    template<typename T, i32 D, i32 Size, i32... Sizes>
    struct Stats : public std::array<Stats<T, D, Sizes...>, Size> {
        using stats = Stats<T, D, Size, Sizes...>;

        void fill(const T& v) {

            // For standard-layout 'this' points to the first struct member
            assert(std::is_standard_layout_v<stats>);

            using entry = StatsEntry<T, D>;
            entry* p = reinterpret_cast<entry*>(this);
            std::fill(p, p + sizeof(*this) / sizeof(entry), v);
        }
    };

    template<typename T, i32 D, i32 Size>
    struct Stats<T, D, Size> : public std::array<StatsEntry<T, D>, Size> {};

    using MainHistoryT = Stats<i16, 16384, 2, 64 * 64>;
    using CaptureHistoryT = Stats<i16, 16384, 2, 6, 64, 6>;
    using PlyHistoryT = Stats<i16, LowPlyClamp, LowPlyCount, 64 * 64>;
    using PieceToHistory = Stats<int16_t, 16384, 12, 64>;
    using ContinuationHistoryT = Stats<PieceToHistory, 0, 12, 64>;
    using CorrectionT = Util::NDArray<i16, 2, 16384>;

    struct HistoryTable {
    public:
        ContinuationHistoryT Continuations[2][2];
        MainHistoryT MainHistory{};
        CaptureHistoryT CaptureHistory{};
        PlyHistoryT PlyHistory{};
        CorrectionT PawnCorrection{};
        CorrectionT NonPawnCorrection{};

        void Clear() {
            MainHistory.fill(0);
            CaptureHistory.fill(0);
            PlyHistory.fill(0);
            std::memset(&PawnCorrection, 0, sizeof(PawnCorrection));
            std::memset(&NonPawnCorrection, 0, sizeof(NonPawnCorrection));

            for (size_t i = 0; i < 2; i++)
            {
                for (size_t j = 0; j < 2; j++)
                {
                    for (auto& to : Continuations[i][j])
                        for (auto& h : to)
                            h->fill(-50);
                }
            }
		}
    };

}


#endif