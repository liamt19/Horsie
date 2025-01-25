#pragma once

#include "defs.h"
#include "move.h"

#include <array>

using Horsie::Move;

namespace Horsie::Cuckoo {

    extern std::array<u64, 8192> keys;
    extern std::array<Move, 8192> moves;

    constexpr auto hash1(u64 key) { return static_cast<i32>(key & 0x1FFF); }
    constexpr auto hash2(u64 key) { return static_cast<i32>((key >> 16) & 0x1FFF); }

    void init();
}
