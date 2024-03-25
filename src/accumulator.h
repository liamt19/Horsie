#pragma once

#ifndef ACCUMULATOR_H
#define ACCUMULATOR_H

#include "types.h"

constexpr int ByteSize = 1536 * sizeof(short);

namespace Horsie {

    struct alignas(64) Accumulator {
    public:
        short White[1536] = {};
        short Black[1536] = {};

        constexpr short* operator[](int c) { return c == 0 ? White : Black; }

        void CopyTo(Accumulator* target) const
        {
            CopyBlock(target->White, White, ByteSize);
            CopyBlock(target->Black, Black, ByteSize);
        }
    };

}

#endif