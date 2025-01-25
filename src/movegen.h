#pragma once

#include "defs.h"
#include "move.h"

namespace Horsie {

    class Position;

    enum MoveGenType {
        PseudoLegal,
        GenNoisy,
        GenEvasions,
        GenNonEvasions,
        GenLegal
    };

    template<MoveGenType>
    i32 Generate(const Position& pos, ScoredMove* moveList, i32 size);
    i32 GenerateQS(const Position& pos, ScoredMove* moveList, i32 size);
}
