#pragma once

#include "../bitboard.h"
#include "../defs.h"
#include "../types.h"
#include "../util/NDArray.h"
#include "arch.h"
#include "network_update.h"

#include <array>
#include <vector>

namespace Horsie {
    class Position;
    class Move;
}

namespace Horsie::NNUE {

    struct alignas(64) Accumulator {
        Util::NDArray<i16, 2, L1_SIZE> Sides{};
        NetworkUpdate Update{};
        std::array<bool, 2> NeedsRefresh = { true, true };
        std::array<bool, 2> Computed = { false, false };

        const std::array<i16, L1_SIZE> operator[](const i32 c) { return Sides[c]; }

        void CopyTo(Accumulator* target) const {
            target->Sides = Sides;
            target->NeedsRefresh = NeedsRefresh;
        }

        void CopyTo(Accumulator* target, const i32 c) const {
            target->Sides[c] = Sides[c];
            target->NeedsRefresh[c] = NeedsRefresh[c];
        }

        void CopyTo(Accumulator& target) const {
            target.Sides = Sides;
            target.NeedsRefresh = NeedsRefresh;
        }

        void CopyTo(Accumulator& target, const i32 c) const {
            target.Sides[c] = Sides[c];
            target.NeedsRefresh[c] = NeedsRefresh[c];
        }

        void MarkDirty() {
            NeedsRefresh[WHITE] = NeedsRefresh[BLACK] = true;
            Computed[WHITE] = Computed[BLACK] = false;
        }
    };


    class AccumulatorStack {
    public:
        AccumulatorStack() : AccStack(256) {
            Reset();
        }
        
        const std::array<i16, L1_SIZE> operator[](const i32 c) { return CurrentAccumulator->Sides[c]; }
        
        void Reset() { 
            HeadIndex = 0;
            CurrentAccumulator = &AccStack[HeadIndex];
            CurrentAccumulator->MarkDirty();
        }

        constexpr Accumulator* Prev() { return &AccStack[HeadIndex - 1]; }
        constexpr Accumulator* Head() { return CurrentAccumulator; }
        constexpr Accumulator* Next() { return &AccStack[HeadIndex + 1]; }

        void MoveNext();
        void MakeMove(const Position& pos, Move m);
        void UndoMove();

        void EnsureUpdated(Position& pos);
        void RefreshIntoCache(Position& pos);
        void RefreshIntoCache(Position& pos, i32 perspective);
        void RefreshFromCache(Position& pos, i32 perspective);

    private:
        std::vector<Accumulator> AccStack{};
        i32 HeadIndex{};
        Accumulator* CurrentAccumulator{};

        void ProcessUpdate(Accumulator* prev, Accumulator* curr, i32 perspective);
    };

    struct FinnyTable {
        Accumulator accumulator;
        std::array<Horsie::Bitboard, 2> Boards = {};
    };

    using BucketCache = std::array<FinnyTable, NNUE::INPUT_BUCKETS * 2>;
}
