#pragma once

#include "../defs.h"
#include "arch.h"
#include "network_update.h"

#include <array>

constexpr i32 ByteSize = L1_SIZE * sizeof(i16);

namespace Horsie {

    struct alignas(64) Accumulator {
    public:
        std::array<std::array<i16, L1_SIZE>, 2> Sides{};
        std::array<bool, 2> NeedsRefresh = { true, true };
        std::array<bool, 2> Computed = { false, false };
        NetworkUpdate Update{};

        const std::array<i16, L1_SIZE> operator[](const i32 c) { return Sides[c]; }

        void CopyTo(Accumulator* target) const {
            target->Sides = Sides;
            target->NeedsRefresh = NeedsRefresh;
        }

        void CopyTo(Accumulator* target, const i32 c) const {
            target->Sides[c] = Sides[c];
            target->NeedsRefresh[c] = NeedsRefresh[c];
        }
    };

}
