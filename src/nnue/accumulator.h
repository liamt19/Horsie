#pragma once

#ifndef ACCUMULATOR_H
#define ACCUMULATOR_H

#include "../types.h"
#include "arch.h"

#include <array>

constexpr int ByteSize = HiddenSize * sizeof(short);

namespace Horsie {

    struct alignas(64) Accumulator {
    public:
        std::array<std::array<short, HiddenSize>, 2> Sides{};
        std::array<bool, 2> NeedsRefresh{};

        const std::array<short, HiddenSize> operator[](const int c) { return Sides[c]; }

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