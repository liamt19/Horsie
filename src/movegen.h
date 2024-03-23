
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


#ifdef MOVEGEN_H_TEMPLATES
template <Color, MoveGenType>
int GenPawns(const Position& pos, ScoredMove* list, ulong targets, int size);

template <Color, MoveGenType>
int GenAll(const Position& pos, ScoredMove* list, int size);

template <MoveGenType>
int GenNormal(const Position& pos, ScoredMove* list, int pt, bool checks, ulong targets, int size);

template <MoveGenType>
int MakePromotionChecks(ScoredMove* list, int from, int promotionSquare, bool isCapture, int size);

int GenCastlingMoves(const Position& pos, ScoredMove* list, int size);

#endif

template<MoveGenType>
int Generate(const Position& pos, ScoredMove* moveList, int size);




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
int GenPawns(const Position& pos, ScoredMove* list, ulong targets, int size);

template <MoveGenType>
int GenAll(const Position& pos, ScoredMove* list, int size);

int GenNormal(const Position& pos, ScoredMove* list, int pt, bool checks, ulong targets, int size);

template <MoveGenType>
int MakePromotionChecks(ScoredMove* list, int from, int promotionSquare, bool isCapture, int size);

int GenCastlingMoves(const Position& pos, ScoredMove* list, int size);

#endif

template<MoveGenType>
int GenAllPseudoLegalMoves(const Position& pos, ScoredMove* moveList);




}


#endif

*/