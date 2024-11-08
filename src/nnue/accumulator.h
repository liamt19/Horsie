#pragma once

#ifndef ACCUMULATOR_H
#define ACCUMULATOR_H

#include "../types.h"
#include <array>

constexpr int ByteSize = 1536 * sizeof(short);

namespace Horsie {

    struct alignas(64) Accumulator {
    public:
        std::array<std::array<short, 1536>, 2> Sides;
        std::array<bool, 2> NeedsRefresh;

        const std::array<short, 1536> operator[](const int c) { return Sides[c]; }

        void CopyTo(Accumulator* target) const
        {
            target->Sides = Sides;
            target->NeedsRefresh = NeedsRefresh;
        }

        void CopyTo(Accumulator* target, const int c) const
        {
            target->Sides[c] = Sides[c];
            target->NeedsRefresh[c] = NeedsRefresh[c];
        }
    };

}

#endif