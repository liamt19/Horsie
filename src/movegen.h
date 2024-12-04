
#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "position.h"
#include "types.h"
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


#endif

















/*


#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "position.h"
#include "types.h"
#include "move.h"


namespace Horsie {

    class Position;

    enum MoveGenType {
        GenLoud,
        GenQuiets,
        GenQChecks,
        GenEvasions,
        GenNonEvasions,
        GenLegal
    };


#ifndef MOVEGEN_H_TEMPLATES
template <MoveGenType>
i32 GenPawns(const Position& pos, ScoredMove* list, u64 targets, i32 size);

template <MoveGenType>
i32 GenAll(const Position& pos, ScoredMove* list, i32 size);

i32 GenNormal(const Position& pos, ScoredMove* list, i32 pt, bool checks, u64 targets, i32 size);

template <MoveGenType>
i32 MakePromotionChecks(ScoredMove* list, i32 from, i32 promotionSquare, bool isCapture, i32 size);

i32 GenCastlingMoves(const Position& pos, ScoredMove* list, i32 size);

#endif

template<MoveGenType>
i32 GenAllPseudoLegalMoves(const Position& pos, ScoredMove* moveList);




}


#endif

*/