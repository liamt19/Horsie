#pragma once

#ifndef CUCKOO_H
#define CUCKOO_H


#include <array>

#include "types.h"
#include "move.h"
using Horsie::Move;

namespace Horsie::Cuckoo {

	extern std::array<ulong, 8192> keys;
	extern std::array<Move, 8192> moves;

	constexpr auto hash1(ulong key) { return static_cast<nuint>(key & 0x1FFF); }
	constexpr auto hash2(ulong key) { return static_cast<nuint>((key >> 16) & 0x1FFF); }

	void init();
}


#endif