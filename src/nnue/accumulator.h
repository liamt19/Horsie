#pragma once

#include "../defs.h"
#include "../types.h"
#include "../util/NDArray.h"
#include "arch.h"
#include "network_update.h"

#include <array>

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
    };

    struct FinnyTable {
        Accumulator accumulator;
        Horsie::Bitboard Boards[2];
    };

    using BucketCache = std::array<FinnyTable, NNUE::INPUT_BUCKETS * 2>;
}
