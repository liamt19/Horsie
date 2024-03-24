#pragma once

#ifndef ACCUMULATOR_H
#define ACCUMULATOR_H

#include "types.h"

constexpr int ByteSize = 1536 * sizeof(short);

namespace Horsie {

    class Accumulator {
    public:

        short* White;
        short* Black;

        Accumulator() {
            White = (short*)AlignedAllocZeroed(ByteSize, AllocAlignment);
            Black = (short*)AlignedAllocZeroed(ByteSize, AllocAlignment);
        }

        constexpr short* operator[](int c) { return c == 0 ? White : Black; }

        void CopyTo(Accumulator* target) const
        {
            CopyBlock(target->White, White, ByteSize);
            CopyBlock(target->Black, Black, ByteSize);
        }

        ~Accumulator() {
            AlignedFree(White);
            AlignedFree(Black);
        }
    };

}

#endif