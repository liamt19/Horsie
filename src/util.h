#pragma once

#ifndef UTIL_H
#define UTIL_H

#include <cstdint>
#include <algorithm>
#include <string>

#include "types.h"
#include "enums.h"

using ulong = uint64_t;


namespace Horsie {

    constexpr int NormalListCapacity = 128;
    constexpr const int MoveListSize = 256;

    constexpr const int MaxDepth = 64;
    constexpr const int MaxPly = 256;

    constexpr std::string_view PieceToChar("pnbrqk");

    inline int FenToPiece(char fenChar)
    {
        fenChar = char(tolower(fenChar));
        switch (fenChar)
        {
        case 'p': return PAWN;
        case 'n': return HORSIE;
        case 'b': return BISHOP;
        case 'r': return ROOK;
        case 'q': return QUEEN;
        case 'k': return KING;
        default: return NONE;
        };
    }

    constexpr int GetPieceValue(int pt) {
        switch (pt) {
        case PAWN: return 199;
        case HORSIE: return 920;
        case BISHOP: return 1058;
        case ROOK: return 1553;
        case QUEEN: return 3127;
        default: return 0;
        }
    }

}

#endif // !UTIL_H