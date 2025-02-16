#pragma once

#include "defs.h"
#include "search_options.h"
#include "types.h"
#include "util/NDArray.h"

#include <array>
#include <cmath>
#include <cstring>

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
            entry += (bonus - (entry * std::abs(bonus) / ClampVal));
        }
    };

    template<typename T, i32 D, i32 Size, i32... Sizes>
    struct Stats : public std::array<Stats<T, D, Sizes...>, Size> {
        using stats = Stats<T, D, Size, Sizes...>;

        void Fill(const T& v) {
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
    using PieceToHistory = Stats<i16, 16384, 12, 64>;
    using ContinuationHistoryT = Stats<PieceToHistory, 0, 12, 64>;
    using CorrectionT = Util::NDArray<i16, 2, 16384>;
    using L2ActivationT = Util::NDArray<i16, 2, 16>;
    using L3ActivationT = Util::NDArray<i16, 2, 32>;

    struct HistoryTable {
    public:
        ContinuationHistoryT Continuations[2][2];
        MainHistoryT MainHistory{};
        CaptureHistoryT CaptureHistory{};
        PlyHistoryT PlyHistory{};
        CorrectionT PawnCorrection{};
        CorrectionT NonPawnCorrection{};
        L2ActivationT L2Correction{};
        L3ActivationT L3Correction{};

        void Clear() {
            MainHistory.Fill(0);
            CaptureHistory.Fill(0);
            PlyHistory.Fill(0);
            std::memset(&PawnCorrection, 0, sizeof(PawnCorrection));
            std::memset(&NonPawnCorrection, 0, sizeof(NonPawnCorrection));
            std::memset(&L2Correction, 0, sizeof(L2Correction));
            std::memset(&L3Correction, 0, sizeof(L3Correction));

            for (size_t i = 0; i < 2; i++) {
                for (size_t j = 0; j < 2; j++) {
                    for (auto& to : Continuations[i][j])
                        for (auto& h : to)
                            h->Fill(-50);
                }
            }
        }

        constexpr auto GetL2Correction(i32 stm, u64 hash) const {
            i32 c = 0;
            u64 h = hash & L2Mask;
            while (h != 0)
                c += L2Correction[stm][poplsb(h)];

            return c / CorrectionGrain;
        }

        constexpr auto GetL3Correction(i32 stm, u64 hash) const {
            i32 c = 0;
            u64 h = hash & L3Mask;
            while (h != 0)
                c += L3Correction[stm][poplsb(h)];

            return c / CorrectionGrain;
        }

        constexpr void UpdateL2Correction(i32 stm, u64 hash, i32 diff, i32 scaledWeight) {
            const auto blendOld = (diff * CorrectionGrain * scaledWeight);
            const auto blendNew = (CorrectionScale - scaledWeight);
            
            u64 h = hash & L2Mask;
            while (h != 0) {
                auto& c = L2Correction[stm][poplsb(h)];
                c = std::clamp((c * blendNew + blendOld) / CorrectionScale, -CorrectionMax, CorrectionMax);
            }
        }

        constexpr void UpdateL3Correction(i32 stm, u64 hash, i32 diff, i32 scaledWeight) {
            const auto blendOld = (diff * CorrectionGrain * scaledWeight);
            const auto blendNew = (CorrectionScale - scaledWeight);

            u64 h = hash & L3Mask;
            while (h != 0) {
                auto& c = L2Correction[stm][poplsb(h)];
                c = std::clamp((c * blendNew + blendOld) / CorrectionScale, -CorrectionMax, CorrectionMax);
            }
        }

    private:
        const u64 L2Mask = 0xFFFF;
        const u64 L3Mask = 0xFFFFFFFF0000;
    };

}
