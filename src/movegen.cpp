
#include "movegen.h"

#include "bitboard.h"
#include "enums.h"
#include "position.h"
#include "precomputed.h"
#include "types.h"


namespace Horsie {

    i32 AddPromotions(ScoredMove* list, i32 from, i32 promotionSquare, i32 size) {
        list[size++].move = Move(from, promotionSquare, FlagPromoQueen);
        list[size++].move = Move(from, promotionSquare, FlagPromoHorsie);
        list[size++].move = Move(from, promotionSquare, FlagPromoRook);
        list[size++].move = Move(from, promotionSquare, FlagPromoBishop);
        return size;
    }


    template <MoveGenType GenType>
    i32 GenPawns(const Position& pos, ScoredMove* list, u64 targets, i32 size) {
        constexpr bool noisyMoves = (GenType & GenNoisy);
        constexpr bool quietMoves = (GenType & GenQuiet);

        const auto stm = pos.ToMove;
        const Bitboard& bb = pos.bb;

        const Direction up = static_cast<Direction>(ShiftUpDir(stm));
        const Direction UpLeft = (up + Direction::WEST);
        const Direction UpRight = (up + Direction::EAST);

        const u64 rank7 = (stm == WHITE) ? Rank7BB : Rank2BB;
        const u64 rank2 = (stm == WHITE) ? Rank2BB : Rank7BB;
        
        const u64 ourPawns = bb.Colors[stm] & bb.Pieces[Piece::PAWN];
        const u64 promotingPawns = ourPawns & rank7;
        const u64 notPromotingPawns = ourPawns & ~rank7;
        const u64 doublePushPawns = ourPawns & rank2;

        const u64 pushSquares = targets & (quietMoves ? ~bb.Occupancy : 0);
        const u64 capSquares = targets & (noisyMoves ? bb.Colors[Not(stm)] : 0);

        if (quietMoves) {
            u64 moves = Forward(stm, notPromotingPawns) & pushSquares;
            u64 twoMoves = DoublePush(stm, doublePushPawns, bb.Occupancy) & pushSquares;
            while (moves != 0) {
                i32 to = poplsb(moves);
                list[size++].move = Move(to - up, to);
            }

            while (twoMoves != 0) {
                i32 to = poplsb(twoMoves);
                list[size++].move = Move(to - up - up, to);
            }
        }

        if (promotingPawns != 0) {
            u64 promotions = Shift(up, promotingPawns) & pushSquares;
            while (promotions != 0) {
                i32 to = poplsb(promotions);
                size = AddPromotions(list, to - up, to, size);
            }

            u64 promoCapsL = Shift(UpLeft, promotingPawns) & capSquares;
            while (promoCapsL != 0) {
                i32 to = poplsb(promoCapsL);
                size = AddPromotions(list, to - up - Direction::WEST, to, size);
            }

            u64 promoCapsR = Shift(UpRight, promotingPawns) & capSquares;
            while (promoCapsR != 0) {
                i32 to = poplsb(promoCapsR);
                size = AddPromotions(list, to - up - Direction::EAST, to, size);
            }
        }

        u64 capturesL = Shift(UpLeft, notPromotingPawns) & capSquares;
        while (capturesL != 0) {
            i32 to = poplsb(capturesL);
            list[size++].move = Move(to - up - Direction::WEST, to);
        }

        u64 capturesR = Shift(UpRight, notPromotingPawns) & capSquares;
        while (capturesR != 0) {
            i32 to = poplsb(capturesR);
            list[size++].move = Move(to - up - Direction::EAST, to);
        }

        if (pos.EPSquare() != EP_NONE && noisyMoves) {
            u64 epMask = notPromotingPawns & PawnAttackMasks[Not(stm)][pos.EPSquare()];
            while (epMask != 0) {
                list[size++].move = Move(poplsb(epMask), pos.EPSquare(), FlagEnPassant);
            }
        }

        return size;
    }

    template <i32 pt>
    i32 GenNormal(const Position& pos, ScoredMove* list, u64 targets, i32 size) {
        const Color stm = pos.ToMove;
        const Bitboard& bb = pos.bb;
        const u64 occ = bb.Occupancy;
        u64 ourPieces = bb.Pieces[pt] & bb.Colors[stm];

        while (ourPieces != 0) {
            i32 idx = poplsb(ourPieces);
            u64 moves = bb.AttackMask<pt>(idx, stm, occ) & targets;
            while (moves != 0) {
                list[size++].move = Move(idx, poplsb(moves));
            }
        }

        return size;
    }

    template i32 Generate<GenQuiet>(const Position&, ScoredMove*, i32);
    template i32 Generate<GenNoisy>(const Position&, ScoredMove*, i32);
    template i32 Generate<GenAll>(const Position&, ScoredMove*, i32);

    template <MoveGenType GenType>
    i32 Generate(const Position& pos, ScoredMove* list, i32 size) {
        constexpr bool noisyMoves = (GenType & GenNoisy);
        constexpr bool quietMoves = (GenType & GenQuiet);

        const Color stm = pos.ToMove;
        const Bitboard& bb = pos.bb;

        const u64 us = bb.Colors[stm];
        const u64 them = bb.Colors[Not(stm)];
        const u64 occ = bb.Occupancy;

        const i32 ourKing = pos.State->KingSquares[stm];

        u64 targets = 0;
        if (noisyMoves) targets |= them;
        if (quietMoves) targets |= ~occ;

        if (pos.InDoubleCheck()) {
            u64 moves = PseudoAttacks[KING][ourKing] & targets;
            while (moves != 0) {
                list[size++].move = Move(ourKing, poplsb(moves));
            }
            return size;
        }

        const u64 kingTargets = targets;
        if (pos.Checked()) {
            targets &= LineBB[ourKing][lsb(pos.State->Checkers)];
        }

        size = GenPawns<GenType>(pos, list, targets, size);
        size = GenNormal<HORSIE>(pos, list, targets, size);
        size = GenNormal<BISHOP>(pos, list, targets, size);
        size = GenNormal<ROOK>(pos, list, targets, size);
        size = GenNormal<QUEEN>(pos, list, targets, size);

        u64 kingMoves = PseudoAttacks[KING][ourKing] & kingTargets;
        while (kingMoves != 0) {
            list[size++].move = Move(ourKing, poplsb(kingMoves));
        }

        if (quietMoves && !pos.Checked()) {
            if (stm == Color::WHITE && (ourKing == static_cast<i32>(Square::E1) || pos.IsChess960)) {
                if (pos.CanCastle(occ, us, CastlingStatus::WK))
                    list[size++].move = Move(ourKing, pos.CastlingRookSquare(CastlingStatus::WK), FlagCastle);

                if (pos.CanCastle(occ, us, CastlingStatus::WQ))
                    list[size++].move = Move(ourKing, pos.CastlingRookSquare(CastlingStatus::WQ), FlagCastle);
            }
            else if (stm == Color::BLACK && (ourKing == static_cast<i32>(Square::E8) || pos.IsChess960)) {
                if (pos.CanCastle(occ, us, CastlingStatus::BK))
                    list[size++].move = Move(ourKing, pos.CastlingRookSquare(CastlingStatus::BK), FlagCastle);

                if (pos.CanCastle(occ, us, CastlingStatus::BQ))
                    list[size++].move = Move(ourKing, pos.CastlingRookSquare(CastlingStatus::BQ), FlagCastle);
            }
        }

        return size;
    }

    i32 GenerateQuiet(const Position& pos, ScoredMove* moveList, i32 size) {
        return Generate<GenQuiet>(pos, moveList, 0);
    }

    i32 GenerateNoisy(const Position& pos, ScoredMove* moveList, i32 size) {
        return Generate<GenNoisy>(pos, moveList, 0);
    }

    i32 GeneratePseudoLegal(const Position& pos, ScoredMove* moveList, i32 size) {
        return Generate<GenAll>(pos, moveList, 0);
    }

    i32 GenerateLegal(const Position& pos, ScoredMove* moveList, i32 size) {
        i32 numMoves = Generate<GenAll>(pos, moveList, 0);

        i32 ourKing = pos.State->KingSquares[pos.ToMove];
        i32 theirKing = pos.State->KingSquares[Not(pos.ToMove)];
        u64 pinned = pos.State->BlockingPieces[pos.ToMove];

        ScoredMove* curr = moveList;
        ScoredMove* end = moveList + numMoves;

        while (curr != end) {
            if (!pos.IsLegal(curr->move, ourKing, theirKing, pinned)) {
                *curr = *--end;
                numMoves--;
            }
            else {
                ++curr;
            }
        }

        return numMoves;
    }

    i32 GenerateQS(const Position& pos, ScoredMove* moveList, i32 size) {
        return pos.Checked() ? Generate<GenAll>(pos, moveList, 0) : Generate<GenNoisy>(pos, moveList, 0);
    }

}
