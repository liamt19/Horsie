#pragma once

#include "defs.h"
#include "move.h"

namespace Horsie {

    class Position;

    enum MoveGenType {
        GenQuiet = 1,
        GenNoisy = 2,
        GenAll = 3,
    };

    template<MoveGenType>
    i32 Generate(const Position& pos, ScoredMove* moveList, i32 size);

    i32 GenerateQuiet(const Position& pos, ScoredMove* moveList, i32 size);
    i32 GenerateNoisy(const Position& pos, ScoredMove* moveList, i32 size);
    i32 GeneratePseudoLegal(const Position& pos, ScoredMove* moveList, i32 size);
    i32 GenerateLegal(const Position& pos, ScoredMove* moveList, i32 size);

    i32 GenerateQS(const Position& pos, ScoredMove* moveList, i32 size);
}
