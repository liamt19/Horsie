
#include "enums.h"
#include "types.h"
#include "position.h"
#include "movegen.h"

#include "precomputed.h"
#include "search.h"

#include <iostream>

//constexpr CastlingStatus operator&(CastlingStatus l, CastlingStatus r) { return l & r; }

namespace Horsie {

namespace {

template <bool noisyMoves>
int MakePromotionChecks(ScoredMove* list, int from, int promotionSquare, bool isCapture, int size)
{
    list[size++].move = Move(from, promotionSquare, FlagPromoQueen);

    if (!noisyMoves || isCapture)
    {
        list[size++].move = Move(from, promotionSquare, FlagPromoKnight);
        list[size++].move = Move(from, promotionSquare, FlagPromoRook);
        list[size++].move = Move(from, promotionSquare, FlagPromoBishop);
    }

    return size;
}


template <MoveGenType GenType>
int GenPawns(const Position& pos, ScoredMove* list, ulong targets, int size) {
    constexpr bool noisyMoves = GenType == GenNoisy;
    constexpr bool evasions = GenType == GenEvasions;
    constexpr bool nonEvasions = GenType == GenNonEvasions;

    const Color stm = pos.ToMove;
    const Bitboard& bb = pos.bb;

    const int theirColor = Not(stm);
    const Direction up = (Direction)ShiftUpDir(stm);
    const Direction UpLeft  = ((Direction)up + Direction::WEST);
    const Direction UpRight = ((Direction)up + Direction::EAST);

    const ulong rank7 = (stm == WHITE) ? Rank7BB : Rank2BB;
    const ulong rank3 = (stm == WHITE) ? Rank3BB : Rank6BB;

    ulong us   = bb.Colors[stm];
    ulong them = bb.Colors[theirColor];
    ulong captureSquares = evasions ? pos.State->Checkers : them;

    ulong emptySquares = ~bb.Occupancy;

    ulong ourPawns = us & bb.Pieces[Piece::PAWN];
    ulong promotingPawns    = ourPawns & rank7;
    ulong notPromotingPawns = ourPawns & ~rank7;

    int theirKing = pos.State->KingSquares[theirColor];

    if (!noisyMoves) {
        //  Include pawn pushes
        ulong moves    = Forward(stm, notPromotingPawns) & emptySquares;
        ulong twoMoves = Forward(stm, moves & rank3)     & emptySquares;

        if (evasions) {
            //  Only include pushes which block the check
            moves    &= targets;
            twoMoves &= targets;
        }

        while (moves != 0) {
            int to = poplsb(moves);
            list[size++].move = Move(to - up, to);
        }

        while (twoMoves != 0) {
            int to = poplsb(twoMoves);
            list[size++].move = Move(to - up - up, to);
        }
    }

    if (promotingPawns != 0) {
        ulong promotions = Shift(up, promotingPawns) & emptySquares;
        ulong promotionCapturesL = Shift(UpLeft, promotingPawns) & captureSquares;
        ulong promotionCapturesR = Shift(UpRight, promotingPawns) & captureSquares;

        if (evasions || noisyMoves) {
            //  Only promote on squares that block the check or capture the checker.
            promotions &= targets;
        }

        while (promotions != 0) {
            int to = poplsb(promotions);
            size = MakePromotionChecks<noisyMoves>(list, to - up, to, false, size);
        }

        while (promotionCapturesL != 0) {
            int to = poplsb(promotionCapturesL);
            size = MakePromotionChecks<noisyMoves>(list, to - up - Direction::WEST, to, true, size);
        }

        while (promotionCapturesR != 0) {
            int to = poplsb(promotionCapturesR);
            size = MakePromotionChecks<noisyMoves>(list, to - up - Direction::EAST, to, true, size);
        }
    }

    ulong capturesL = Shift(UpLeft, notPromotingPawns) & captureSquares;
    ulong capturesR = Shift(UpRight, notPromotingPawns) & captureSquares;

    while (capturesL != 0) {
        int to = poplsb(capturesL);
        list[size++].move = Move(to - up - Direction::WEST, to);
    }

    while (capturesR != 0) {
        int to = poplsb(capturesR);
        list[size++].move = Move(to - up - Direction::EAST, to);
    }

    if (pos.State->EPSquare != EP_NONE && !noisyMoves) {
        if (evasions && (targets & (SquareBB(pos.State->EPSquare + up))) != 0) {
            //  When in check, we can only en passant if the pawn being captured is the one giving check
            return size;
        }

        ulong mask = notPromotingPawns & PawnAttackMasks[theirColor][pos.State->EPSquare];
        while (mask != 0) {
            int from = poplsb(mask);
            list[size++].move = Move(from, pos.State->EPSquare, FlagEnPassant);
        }
    }

    return size;
}


int GenNormal(const Position& pos, ScoredMove* list, int pt, ulong targets, int size) {
    const Color stm = pos.ToMove;
    const Bitboard& bb = pos.bb;
    ulong occ = bb.Occupancy;
    ulong ourPieces = bb.Pieces[pt] & bb.Colors[stm];

    while (ourPieces != 0) {
        int idx = poplsb(ourPieces);
        ulong moves = bb.AttackMask(idx, stm, pt, occ) & targets;

        while (moves != 0) {
            int to = poplsb(moves);
            list[size++].move = Move(idx, to);
        }
    }

    return size;
}

template <MoveGenType GenType>
int GenAll(const Position& pos, ScoredMove* list, int size) {
    constexpr bool noisyMoves = GenType == GenNoisy;
    constexpr bool evasions = GenType == GenEvasions;
    constexpr bool nonEvasions = GenType == GenNonEvasions;

    const Color stm = pos.ToMove;
    const Bitboard& bb = pos.bb;

    ulong us   = bb.Colors[stm];
    ulong them = bb.Colors[Not(stm)];
    ulong occ  = bb.Occupancy;

    int ourKing   = pos.State->KingSquares[stm];
    int theirKing = pos.State->KingSquares[Not(stm)];

    ulong targets = 0;

    // If we are generating evasions and in double check, then skip non-king moves.
    if (!(evasions && MoreThanOne(pos.State->Checkers)))
    {
        targets = evasions    ? LineBB[ourKing][lsb(pos.State->Checkers)]
                : nonEvasions ? ~us
                : noisyMoves  ?  them
                :               ~occ;

        size = GenPawns<GenType>(pos, list, targets, size);
        size = GenNormal(pos, list, HORSIE, targets, size);
        size = GenNormal(pos, list, BISHOP, targets, size);
        size = GenNormal(pos, list, ROOK, targets, size);
        size = GenNormal(pos, list, QUEEN, targets, size);
    }

    ulong moves = PseudoAttacks[KING][ourKing] & (evasions ? ~us : targets);
    while (moves != 0) {
        int to = poplsb(moves);
        list[size++].move = Move(ourKing, to);
    }

    if (nonEvasions)
    {
        if (stm == Color::WHITE && (ourKing == (int)Square::E1 || pos.IsChess960))
        {
            if (pos.CanCastle(occ, us, CastlingStatus::WK))
                list[size++].move = Move(ourKing, pos.CastlingRookSquares[(int)CastlingStatus::WK], FlagCastle);

            if (pos.CanCastle(occ, us, CastlingStatus::WQ))
                list[size++].move = Move(ourKing, pos.CastlingRookSquares[(int)CastlingStatus::WQ], FlagCastle);
        }
        else if (stm == Color::BLACK && (ourKing == (int)Square::E8 || pos.IsChess960))
        {
            if (pos.CanCastle(occ, us, CastlingStatus::BK))
                list[size++].move = Move(ourKing, pos.CastlingRookSquares[(int)CastlingStatus::BK], FlagCastle);

            if (pos.CanCastle(occ, us, CastlingStatus::BQ))
                list[size++].move = Move(ourKing, pos.CastlingRookSquares[(int)CastlingStatus::BQ], FlagCastle);
        }
    }

    return size;
}


}

template <MoveGenType GenType>
int Generate(const Position& pos, ScoredMove* moveList, int size) {
    return pos.State->Checkers ? GenAll<GenEvasions>   (pos, moveList, 0) : 
                                 GenAll<GenNonEvasions>(pos, moveList, 0);
}



int GenerateQS(const Position& pos, ScoredMove* moveList, int size) {
    return pos.State->Checkers ? GenAll<GenEvasions>(pos, moveList, 0) :
                                 GenAll<GenNoisy>   (pos, moveList, 0);
}


template int Generate<PseudoLegal>(const Position&, ScoredMove*, int);
template int Generate<GenNoisy>(const Position&, ScoredMove*, int);
template int Generate<GenEvasions>(const Position&, ScoredMove*, int);
template int Generate<GenNonEvasions>(const Position&, ScoredMove*, int);

template<>
int Generate<GenLegal>(const Position& pos, ScoredMove* moveList, int size) {

    int numMoves = pos.State->Checkers ? Generate<GenEvasions>(pos, moveList, 0) :
                                         Generate<GenNonEvasions>(pos, moveList, 0);

    int ourKing = pos.State->KingSquares[pos.ToMove];
    int theirKing = pos.State->KingSquares[Not(pos.ToMove)];
    ulong pinned = pos.State->BlockingPieces[pos.ToMove];

    ScoredMove* curr = moveList;
    ScoredMove* end = moveList + numMoves;

    while (curr != end)
    {
        if (!pos.IsLegal(curr->move, ourKing, theirKing, pinned))
        {
            *curr = *--end;
            numMoves--;
        }
        else
        {
            ++curr;
        }
    }

    return numMoves;
}


}

