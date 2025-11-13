#pragma once

#include "defs.h"
#include "enums.h"
#include "move.h"
#include "types.h"
#include "util/NDArray.h"

#include <array>
#include <cmath>
#include <cstring>
#include <span>

namespace Horsie {

    constexpr i32 HistoryClamp = 16384;
    constexpr i32 LowPlyCount = 4;
    constexpr i32 LowPlyClamp = 8192;

    constexpr i32 CorrectionSize = 16384;
    constexpr i32 CorrectionClamp = 1024;
    constexpr i32 CorrectionBonusLimit = CorrectionClamp / 4;

    constexpr i32 ContinuationMax = 16384;
    constexpr i16 ContinuationFill = -50;


    struct ContinuationEntry {
        i16 entry;

        ContinuationEntry() = default;
        ContinuationEntry(const i16& v) : entry{ v } {};

        void operator=(const i16& v) { entry = v; }
        i16* operator&() { return &entry; }
        i16* operator->() { return &entry; }
        operator const i16& () const { return entry; }
        void operator<<(i32 bonus) { entry += (bonus - (entry * std::abs(bonus) / ContinuationMax)); }
    };

    template<typename T, i32 ClampVal>
    class StatsEntry {
        T entry;

    public:
        constexpr void operator=(const T& v) { entry = v; }
        constexpr T* operator&() { return &entry; }
        constexpr T* operator->() { return &entry; }
        constexpr operator const T& () const { return entry; }

        void operator<<(i32 bonus) {
            entry += (bonus - (entry * std::abs(bonus) / ClampVal));
        }
    };

    template<typename T, i32 D, i32 Size, i32... Sizes>
    struct Stats : public std::array<Stats<T, D, Sizes...>, Size> {
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

    using CorrectionT = Stats<i16, CorrectionClamp, 2, CorrectionSize>;

    using PieceToHistory = Util::NDArray<ContinuationEntry, 12, 64>;
    using ContinuationHistoryT = Util::NDArray<PieceToHistory, 12, 64>;

    struct alignas(64) HistoryTable {
    public:
        ContinuationHistoryT Continuations[2][2];
        MainHistoryT MainHistory{};
        CaptureHistoryT CaptureHistory{};
        PlyHistoryT PlyHistory{};
        CorrectionT PawnCorrection{};
        CorrectionT NonPawnCorrection{};

        void Clear() {
            MainHistory.Fill(0);
            CaptureHistory.Fill(0);
            PlyHistory.Fill(0);
            std::memset(&PawnCorrection, 0, sizeof(PawnCorrection));
            std::memset(&NonPawnCorrection, 0, sizeof(NonPawnCorrection));

            for (size_t i = 0; i < 2; i++) {
                for (size_t j = 0; j < 2; j++) {
                    for (auto& pt1 : Continuations[i][j])
                        for (auto& sq1 : pt1)
                            for (auto& subArr : sq1) {
                                subArr.fill({ ContinuationFill });
                            }
                }
            }
        }


        void UpdateMainHistory(Color stm, Move move, i32 bonus) { MainHistory[stm][move.GetMoveMask()] << bonus; }
        i16   GetMainHistory(Color stm, Move move) const { return MainHistory[stm][move.GetMoveMask()]; }

        void UpdatePlyHistory(i16 ply, Move move, i32 bonus) { PlyHistory[ply][move.GetMoveMask()] << bonus; }
        i16   GetPlyHistory(i16 ply, Move move) const { return PlyHistory[ply][move.GetMoveMask()]; }

        void UpdateNoisyHistory(Color stm, i32 movingPiece, i32 dstSq, i32 capturedPiece, i32 bonus) { CaptureHistory[stm][movingPiece][dstSq][capturedPiece] << bonus; }
        i16 GetNoisyHistory(Color stm, i32 ourPieceType, i32 moveTo, i32 theirPieceType) const { return CaptureHistory[stm][ourPieceType][moveTo][theirPieceType]; }




        void UpdateQuietScore(std::span<PieceToHistory* const> cont, std::span<Move const> moves, i16 ply, i32 piece, Move move, i32 bonus, bool checked) {
            const auto stm = ColorOfPiece(piece);
            const auto dstSq = move.To();

            UpdateMainHistory(stm, move, bonus);
            if (ply < LowPlyCount)
                UpdatePlyHistory(ply, move, bonus);

            UpdateContinuations(cont, moves, ply, piece, dstSq, bonus, checked);
        }


        void UpdateContinuations(std::span<PieceToHistory* const> cont, std::span<Move const> moves, i16 ply, i32 piece, i32 dstSq, i32 bonus, bool checked) {
            for (auto i : { 1, 2, 4, 6 }) {
                if (ply < i) break;
                if (checked && i > 2) break;

                if (moves[ply - i])
                    (*cont[ply - i])[piece][dstSq] << bonus;
            }
        }

        i16 GetContinuationEntry(std::span<PieceToHistory* const> cont, i16 ply, i32 i, i32 piece, i32 dstSq) const {
            if (ply < i)
                return Continuations[0][0][0][0][0][0]; // Jesus wept

            return (*cont[ply - i])[piece][dstSq];
        }


    };

}
