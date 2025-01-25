
#include "movegen.h"

#include "bitboard.h"
#include "enums.h"
#include "position.h"
#include "precomputed.h"
#include "types.h"


namespace Horsie {

    namespace {

        template <bool noisyMoves>
        i32 MakePromotionChecks(ScoredMove* list, i32 from, i32 promotionSquare, bool isCapture, i32 size) {
            list[size++].move = Move(from, promotionSquare, FlagPromoQueen);

            if (!noisyMoves || isCapture) {
                list[size++].move = Move(from, promotionSquare, FlagPromoHorsie);
                list[size++].move = Move(from, promotionSquare, FlagPromoRook);
                list[size++].move = Move(from, promotionSquare, FlagPromoBishop);
            }

            return size;
        }


        template <MoveGenType GenType>
        i32 GenPawns(const Position& pos, ScoredMove* list, u64 targets, i32 size) {
            constexpr bool noisyMoves = GenType == GenNoisy;
            constexpr bool evasions = GenType == GenEvasions;
            constexpr bool nonEvasions = GenType == GenNonEvasions;

            const Color stm = pos.ToMove;
            const Bitboard& bb = pos.bb;

            const i32 theirColor = Not(stm);
            const Direction up = static_cast<Direction>(ShiftUpDir(stm));
            const Direction UpLeft  = (up + Direction::WEST);
            const Direction UpRight = (up + Direction::EAST);

            const u64 rank7 = (stm == WHITE) ? Rank7BB : Rank2BB;
            const u64 rank3 = (stm == WHITE) ? Rank3BB : Rank6BB;

            const u64 us   = bb.Colors[stm];
            const u64 them = bb.Colors[theirColor];
            const u64 captureSquares = evasions ? pos.State->Checkers : them;

            const u64 emptySquares = ~bb.Occupancy;

            const u64 ourPawns = us & bb.Pieces[Piece::PAWN];
            const u64 promotingPawns    = ourPawns & rank7;
            const u64 notPromotingPawns = ourPawns & ~rank7;

            if (!noisyMoves) {
                //  Include pawn pushes
                u64 moves    = Forward(stm, notPromotingPawns) & emptySquares;
                u64 twoMoves = Forward(stm, moves & rank3)     & emptySquares;

                if (evasions) {
                    //  Only include pushes which block the check
                    moves &= targets;
                    twoMoves &= targets;
                }

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
                u64 promotions = Shift(up, promotingPawns) & emptySquares;
                u64 promotionCapturesL = Shift(UpLeft, promotingPawns) & captureSquares;
                u64 promotionCapturesR = Shift(UpRight, promotingPawns) & captureSquares;

                if (evasions || noisyMoves) {
                    //  Only promote on squares that block the check or capture the checker.
                    promotions &= targets;
                }

                while (promotions != 0) {
                    i32 to = poplsb(promotions);
                    size = MakePromotionChecks<noisyMoves>(list, to - up, to, false, size);
                }

                while (promotionCapturesL != 0) {
                    i32 to = poplsb(promotionCapturesL);
                    size = MakePromotionChecks<noisyMoves>(list, to - up - Direction::WEST, to, true, size);
                }

                while (promotionCapturesR != 0) {
                    i32 to = poplsb(promotionCapturesR);
                    size = MakePromotionChecks<noisyMoves>(list, to - up - Direction::EAST, to, true, size);
                }
            }

            u64 capturesL = Shift(UpLeft, notPromotingPawns) & captureSquares;
            u64 capturesR = Shift(UpRight, notPromotingPawns) & captureSquares;

            while (capturesL != 0) {
                i32 to = poplsb(capturesL);
                list[size++].move = Move(to - up - Direction::WEST, to);
            }

            while (capturesR != 0) {
                i32 to = poplsb(capturesR);
                list[size++].move = Move(to - up - Direction::EAST, to);
            }

            if (pos.State->EPSquare != EP_NONE && !noisyMoves) {
                if (evasions && (targets & (SquareBB(pos.State->EPSquare + up))) != 0) {
                    //  When in check, we can only en passant if the pawn being captured is the one giving check
                    return size;
                }

                u64 mask = notPromotingPawns & PawnAttackMasks[theirColor][pos.State->EPSquare];
                while (mask != 0) {
                    i32 from = poplsb(mask);
                    list[size++].move = Move(from, pos.State->EPSquare, FlagEnPassant);
                }
            }

            return size;
        }

        i32 GenNormal(const Position& pos, ScoredMove* list, i32 pt, u64 targets, i32 size) {
            const Color stm = pos.ToMove;
            const Bitboard& bb = pos.bb;
            const u64 occ = bb.Occupancy;
            u64 ourPieces = bb.Pieces[pt] & bb.Colors[stm];

            while (ourPieces != 0) {
                i32 idx = poplsb(ourPieces);
                u64 moves = bb.AttackMask(idx, stm, pt, occ) & targets;

                while (moves != 0) {
                    i32 to = poplsb(moves);
                    list[size++].move = Move(idx, to);
                }
            }

            return size;
        }

        template <MoveGenType GenType>
        i32 GenAll(const Position& pos, ScoredMove* list, i32 size) {
            constexpr bool noisyMoves = GenType == GenNoisy;
            constexpr bool evasions = GenType == GenEvasions;
            constexpr bool nonEvasions = GenType == GenNonEvasions;

            const Color stm = pos.ToMove;
            const Bitboard& bb = pos.bb;

            u64 us   = bb.Colors[stm];
            u64 them = bb.Colors[Not(stm)];
            u64 occ  = bb.Occupancy;

            i32 ourKing   = pos.State->KingSquares[stm];
            i32 theirKing = pos.State->KingSquares[Not(stm)];

            u64 targets = 0;

            // If we are generating evasions and in double check, then skip non-king moves.
            if (!(evasions && MoreThanOne(pos.State->Checkers))) {
                targets = evasions    ?  LineBB[ourKing][lsb(pos.State->Checkers)]
                        : nonEvasions ? ~us
                        : noisyMoves  ?  them
                        :               ~occ;

                size = GenPawns<GenType>(pos, list, targets, size);
                size = GenNormal(pos, list, HORSIE, targets, size);
                size = GenNormal(pos, list, BISHOP, targets, size);
                size = GenNormal(pos, list, ROOK, targets, size);
                size = GenNormal(pos, list, QUEEN, targets, size);
            }

            u64 moves = PseudoAttacks[KING][ourKing] & (evasions ? ~us : targets);
            while (moves != 0) {
                i32 to = poplsb(moves);
                list[size++].move = Move(ourKing, to);
            }

            if (nonEvasions) {
                if (stm == Color::WHITE && (ourKing == static_cast<i32>(Square::E1) || pos.IsChess960)) {
                    if (pos.CanCastle(occ, us, CastlingStatus::WK))
                        list[size++].move = Move(ourKing, pos.CastlingRookSquares[static_cast<i32>(CastlingStatus::WK)], FlagCastle);

                    if (pos.CanCastle(occ, us, CastlingStatus::WQ))
                        list[size++].move = Move(ourKing, pos.CastlingRookSquares[static_cast<i32>(CastlingStatus::WQ)], FlagCastle);
                }
                else if (stm == Color::BLACK && (ourKing == static_cast<i32>(Square::E8) || pos.IsChess960)) {
                    if (pos.CanCastle(occ, us, CastlingStatus::BK))
                        list[size++].move = Move(ourKing, pos.CastlingRookSquares[static_cast<i32>(CastlingStatus::BK)], FlagCastle);

                    if (pos.CanCastle(occ, us, CastlingStatus::BQ))
                        list[size++].move = Move(ourKing, pos.CastlingRookSquares[static_cast<i32>(CastlingStatus::BQ)], FlagCastle);
                }
            }

            return size;
        }

    }

    template <MoveGenType GenType>
    i32 Generate(const Position& pos, ScoredMove* moveList, i32 size) {
        return pos.State->Checkers ? GenAll<GenEvasions>(pos, moveList, 0) :
                                     GenAll<GenNonEvasions>(pos, moveList, 0);
    }

    i32 GenerateQS(const Position& pos, ScoredMove* moveList, i32 size) {
        return pos.State->Checkers ? GenAll<GenEvasions>(pos, moveList, 0) :
                                     GenAll<GenNoisy>(pos, moveList, 0);
    }

    template i32 Generate<PseudoLegal>(const Position&, ScoredMove*, i32);
    template i32 Generate<GenNoisy>(const Position&, ScoredMove*, i32);
    template i32 Generate<GenEvasions>(const Position&, ScoredMove*, i32);
    template i32 Generate<GenNonEvasions>(const Position&, ScoredMove*, i32);

    template<>
    i32 Generate<GenLegal>(const Position& pos, ScoredMove* moveList, i32 size) {
        i32 numMoves = pos.State->Checkers ? Generate<GenEvasions>(pos, moveList, 0) :
                                             Generate<GenNonEvasions>(pos, moveList, 0);

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
}

