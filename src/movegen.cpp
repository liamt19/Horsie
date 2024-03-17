


#include "position.h"
#include "movegen.h"
#include "types.h"
#include "precomputed.h"


namespace Horsie {

    constexpr CastlingStatus operator&(CastlingStatus l, CastlingStatus r) { return l & r; }

template <MoveGenType GenType>
int GenPawns(const Position& pos, ScoredMove* list, ulong targets, int size) {
    constexpr bool loudMoves = GenType == GenLoud;
    constexpr bool quiets = GenType == GenQuiets;
    constexpr bool quietChecks = GenType == GenQChecks;
    constexpr bool evasions = GenType == GenEvasions;
    constexpr bool nonEvasions = GenType == GenNonEvasions;


    ulong rank7 = (pos.ToMove == WHITE) ? Rank7BB : Rank2BB;
    ulong rank3 = (pos.ToMove == WHITE) ? Rank3BB : Rank6BB;

    constexpr int up = ShiftUpDir(pos.ToMove);

    int theirColor = ~pos.ToMove;

    Bitboard bb = pos.bb;

    ulong us = bb.Colors[pos.ToMove];
    ulong them = bb.Colors[theirColor];
    ulong captureSquares = evasions ? pos.State()->Checkers : them;

    constexpr Direction UpLeft =  ((Direction) up + Direction::WEST);
    constexpr Direction UpRight = ((Direction) up + Direction::EAST);

    ulong occupiedSquares = them | us;
    ulong emptySquares = ~occupiedSquares;

    ulong ourPawns = us & bb.Pieces[Piece::PAWN];
    ulong promotingPawns = ourPawns & rank7;
    ulong notPromotingPawns = ourPawns & ~rank7;

    int theirKing = pos.State->KingSquares[theirColor];

    if (!loudMoves)
    {
        //  Include pawn pushes
        ulong moves = Forward(pos.ToMove, notPromotingPawns) & emptySquares;
        ulong twoMoves = Forward(pos.ToMove, moves & rank3) & emptySquares;

        if (evasions)
        {
            //  Only include pushes which block the check
            moves &= targets;
            twoMoves &= targets;
        }

        if (quietChecks)
        {
            //  Only include pushes that cause a discovered check, or directly check the king
            ulong discoveryPawns = pos.State->BlockingPieces[theirColor] & ~FileBB(theirKing);
            moves &= PawnAttackMasks[theirColor][theirKing] | Shift<up>(discoveryPawns);
            twoMoves &= PawnAttackMasks[theirColor][theirKing] | Shift<up + up>(discoveryPawns);
        }

        while (moves != 0)
        {
            int to = poplsb(moves);
            list[size++].Move.SetNew(to - up, to);
        }

        while (twoMoves != 0)
        {
            int to = poplsb(twoMoves);
            list[size++].Move.SetNew(to - up - up, to);
        }
    }

    if (promotingPawns != 0)
    {
        ulong promotions = Shift<up>(promotingPawns) & emptySquares;

        //ulong promotionCapturesL = Shift((Direction)(up + Direction::WEST), promotingPawns) & captureSquares;
        //ulong promotionCapturesR = Shift((Direction)(up + Direction::EAST), promotingPawns) & captureSquares;

        ulong promotionCapturesL = Shift<UpLeft>(promotingPawns) & captureSquares;
        ulong promotionCapturesR = Shift<UpRight>(promotingPawns) & captureSquares;

        if (evasions)
        {
            //  Only promote on squares that block the check or capture the checker.
            promotions &= targets;
        }

        while (promotions != 0)
        {
            int to = poplsb(promotions);
            size = MakePromotionChecks<GenType>(list, to - up, to, false, size);
        }

        while (promotionCapturesL != 0)
        {
            int to = poplsb(promotionCapturesL);
            size = MakePromotionChecks<GenType>(list, to - up - Direction::WEST, to, true, size);
        }

        while (promotionCapturesR != 0)
        {
            int to = poplsb(promotionCapturesR);
            size = MakePromotionChecks<GenType>(list, to - up - Direction::EAST, to, true, size);
        }
    }

    if (!(quiets || quietChecks))
    {
        //  Don't generate captures for quiets
        //ulong capturesL = Shift((Direction)(up + Direction::WEST), notPromotingPawns) & captureSquares;
        //ulong capturesR = Shift((Direction)(up + Direction::EAST), notPromotingPawns) & captureSquares;

        ulong capturesL = Shift<UpLeft>(notPromotingPawns) & captureSquares;
        ulong capturesR = Shift<UpRight>(notPromotingPawns) & captureSquares;

        while (capturesL != 0)
        {
            int to = poplsb(capturesL);
            list[size++].Move.SetNew(to - up - Direction::WEST, to);
        }

        while (capturesR != 0)
        {
            int to = poplsb(capturesR);
            list[size++].Move.SetNew(to - up - Direction::EAST, to);
        }

        if (pos.State->EPSquare != EP_NONE)
        {
            if (evasions && (targets & (SquareBB(pos.State->EPSquare + up))) != 0)
            {
                //  When in check, we can only en passant if the pawn being captured is the one giving check
                return size;
            }

            ulong mask = notPromotingPawns & PawnAttackMasks[theirColor][pos.State->EPSquare];
            while (mask != 0)
            {
                int from = poplsb(mask);

                list[size++].Move.SetNew(from, pos.State->EPSquare);
                list[size - 1].Move.SetEnPassant();
            }
        }
    }

    return size;
}


template <MoveGenType GenType>
int GenAll(const Position& pos, ScoredMove* list, int size) {
    constexpr bool loudMoves = GenType == GenLoud;
    constexpr bool quiets = GenType == GenQuiets;
    constexpr bool quietChecks = GenType == GenQChecks;
    constexpr bool evasions = GenType == GenEvasions;
    constexpr bool nonEvasions = GenType == GenNonEvasions;

    Bitboard bb = pos.bb;
    ulong us = bb.Colors[pos.ToMove];
    ulong them = bb.Colors[~pos.ToMove];
    ulong occ = us | them;

    int ourKing = pos.State->KingSquares[pos.ToMove];
    int theirKing = pos.State->KingSquares[~pos.ToMove];

    ulong targets = 0;

    // If we are generating evasions and in double check, then skip non-king moves.
    if (!(evasions && MoreThanOne(pos.State->Checkers)))
    {
        targets = evasions ? LineBB[ourKing][lsb(pos.State->Checkers)]
                : nonEvasions ? ~us
                : loudMoves ? them
                : ~occ;

        size = GenPawns<GenType>(pos, list, targets, size);
        size = GenNormal(pos, list, Knight, quietChecks, targets, size);
        size = GenNormal(pos, list, Bishop, quietChecks, targets, size);
        size = GenNormal(pos, list, Rook, quietChecks, targets, size);
        size = GenNormal(pos, list, Queen, quietChecks, targets, size);
    }

    //  If we are doing non-captures with check and our king isn't blocking a check, then skip generating king moves
    if (!(quietChecks && (pos.State->BlockingPieces[~pos.ToMove] & SquareBB(ourKing)) == 0))
    {
        ulong moves = PseudoAttacks[KING][ourKing] & (evasions ? ~us : targets);
        if (quietChecks)
        {
            //  If we ARE doing non-captures with check and our king is a blocker,
            //  then only generate moves that get the king off of any shared ranks/files.
            //  Note we can't move our king from one shared ray to another since we can only move diagonally 1 square
            //  and their king would be attacking ours.
            moves &= ~bb.AttackMask(theirKing, ~pos.ToMove, QUEEN, occ);
        }

        while (moves != 0)
        {
            int to = poplsb(moves);
            list[size++].Move.SetNew(ourKing, to);
        }

        if ((quiets || nonEvasions) && ((pos.State->CastleStatus & (pos.ToMove == WHITE ? CastlingStatus::White : CastlingStatus::Black)) != CastlingStatus::None))
        {
            //  Only do castling moves if we are doing non-captures or we aren't in check.
            size = GenCastlingMoves(pos, list, size);
        }
    }

    return size;
}


int GenNormal(const Position& pos, ScoredMove* list, int pt, bool checks, ulong targets, int size)
{
    // TODO: JIT seems to prefer having separate methods for each piece type, instead of a 'pt' parameter
    // This is far more convenient though

    Bitboard bb = pos.bb;
    ulong us = bb.Colors[pos.ToMove];
    ulong them = bb.Colors[~pos.ToMove];
    ulong occ = us | them;

    ulong ourPieces = bb.Pieces[pt] & bb.Colors[pos.ToMove];
    while (ourPieces != 0)
    {
        int idx = poplsb(ourPieces);
        ulong moves = bb.AttackMask(idx, pos.ToMove, pt, occ) & targets;

        if (checks && (pt == QUEEN || ((pos.State->BlockingPieces[~pos.ToMove] & SquareBB(idx)) == 0)))
        {
            moves &= pos.State->CheckSquares[pt];
        }

        while (moves != 0)
        {
            int to = poplsb(moves);
            list[size++].Move.SetNew(idx, to);
        }
    }

    return size;
}



template <MoveGenType GenType>
int MakePromotionChecks(ScoredMove* list, int from, int promotionSquare, bool isCapture, int size) {
    constexpr bool loudMoves = GenType == GenLoud;
    constexpr bool quiets = GenType == GenQuiets;
    constexpr bool quietChecks = GenType == GenQChecks;
    constexpr bool evasions = GenType == GenEvasions;
    constexpr bool nonEvasions = GenType == GenNonEvasions;
    
    int highPiece = Knight;
    int lowPiece = Queen;

    if (quiets && isCapture)
    {
        return size;
    }

    if (evasions || nonEvasions)
    {
        //  All promotions are valid
        highPiece = Queen;
        lowPiece = Knight;
    }

    if (loudMoves)
    {
        //  GenLoud makes all promotions for captures, and only makes Queens for non-capture promotions
        highPiece = Queen;
        lowPiece = isCapture ? Knight : Queen;
    }

    if (quiets)
    {
        //  GenQuiets only makes underpromotions
        lowPiece = Knight;
        highPiece = Rook;
    }

    //  GenQChecks makes nothing.

    for (int promotionPiece = lowPiece; promotionPiece <= highPiece; promotionPiece++)
    {
        list[size++].Move.SetNew(from, promotionSquare, promotionPiece);
    }

    return size;
}


int GenCastlingMoves(const Position& pos, ScoredMove* list, int size)
{
    Bitboard bb = pos.bb;
    ulong occ = bb.Occupancy;
    ulong us = bb.Colors[pos.ToMove];
    int ourKing = pos.State->KingSquares[pos.ToMove];
    
    if (pos.ToMove == WHITE && (ourKing == (int)Square::E1 || pos.IsChess960))
    {
        if ((pos.State->CastleStatus & CastlingStatus::WK) != CastlingStatus::None
            && (occ & pos.CastlingRookPaths[(int)CastlingStatus::WK]) == 0
            && (bb.Pieces[Rook] & SquareBB(pos.CastlingRookSquares[(int)CastlingStatus::WK]) & us) != 0)
        {
            list[size++].Move.SetNew(ourKing, pos.CastlingRookSquares[(int)CastlingStatus::WK]);
            list[size - 1].Move.SetCastle();
        }

        if ((pos.State->CastleStatus & CastlingStatus::WQ) != CastlingStatus::None
            && (occ & pos.CastlingRookPaths[(int)CastlingStatus::WQ]) == 0
            && (bb.Pieces[Rook] & SquareBB(pos.CastlingRookSquares[(int)CastlingStatus::WQ]) & us) != 0)
        {
            list[size++].Move.SetNew(ourKing, pos.CastlingRookSquares[(int)CastlingStatus::WQ]);
            list[size - 1].Move.SetCastle();
        }
    }
    else if (pos.ToMove == BLACK && (ourKing == (int)Square::E8 || pos.IsChess960))
    {
        if ((pos.State->CastleStatus & CastlingStatus::BK) != CastlingStatus::None
            && (occ & pos.CastlingRookPaths[(int)CastlingStatus::BK]) == 0
            && (bb.Pieces[Rook] & SquareBB(pos.CastlingRookSquares[(int)CastlingStatus::BK]) & us) != 0)
        {
            list[size++].Move.SetNew(ourKing, pos.CastlingRookSquares[(int)CastlingStatus::BK]);
            list[size - 1].Move.SetCastle();
        }

        if ((pos.State->CastleStatus & CastlingStatus::BQ) != CastlingStatus::None
            && (occ & pos.CastlingRookPaths[(int)CastlingStatus::BQ]) == 0
            && (bb.Pieces[Rook] & SquareBB(pos.CastlingRookSquares[(int)CastlingStatus::BQ]) & us) != 0)
        {
            list[size++].Move.SetNew(ourKing, pos.CastlingRookSquares[(int)CastlingStatus::BQ]);
            list[size - 1].Move.SetCastle();
        }
    }

    return size;
}

}

