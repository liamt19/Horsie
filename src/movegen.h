
#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "position.h"
#include "types.h"
#include "move.h"


namespace Horsie {
    enum MoveGenType {
    GenLoud,
    GenQuiets,
    GenQChecks,
    GenEvasions,
    GenNonEvasions,
};

template <MoveGenType GenType>
int GenPawns(const Position& pos, ScoredMove* list, ulong targets, int size);

template <MoveGenType GenType>
int GenAll(const Position& pos, ScoredMove* list, int size);

int GenNormal(const Position& pos, ScoredMove* list, int pt, bool checks, ulong targets, int size);

template <MoveGenType GenType>
int MakePromotionChecks(ScoredMove* list, int from, int promotionSquare, bool isCapture, int size);

int GenCastlingMoves(const Position& pos, ScoredMove* list, int size);


}


#endif