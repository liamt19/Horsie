
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
            const u64 captureSquares = evasions ? pos.Checkers() : them;

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

            const auto epSq = pos.EPSquare();
            if (epSq != EP_NONE && !noisyMoves) {
                if (evasions && (targets & (SquareBB(epSq + up))) != 0) {
                    //  When in check, we can only en passant if the pawn being captured is the one giving check
                    return size;
                }

                u64 mask = notPromotingPawns & PawnAttackMasks[theirColor][epSq];
                while (mask != 0) {
                    i32 from = poplsb(mask);
                    list[size++].move = Move(from, epSq, FlagEnPassant);
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

            i32 ourKing = pos.KingSquare(stm);
            u64 targets = 0;

            // If we are generating evasions and in double check, then skip non-king moves.
            if (!(evasions && MoreThanOne(pos.Checkers()))) {
                targets = evasions    ?  LineBB[ourKing][lsb(pos.Checkers())]
                        : nonEvasions ? ~us
                        : noisyMoves  ?  them
                        :               ~occ;

                size = GenPawns<GenType>(pos, list, targets, size);
                size = GenNormal<HORSIE>(pos, list, targets, size);
                size = GenNormal<BISHOP>(pos, list, targets, size);
                size = GenNormal<ROOK>(pos, list, targets, size);
            }

            u64 moves = PseudoAttacks[KING][ourKing] & (evasions ? ~us : targets);
            moves &= ~pos.GetAllThreats();

            while (moves != 0) {
                i32 to = poplsb(moves);
                list[size++].move = Move(ourKing, to);
            }
            
            if (nonEvasions) {
                const auto col = (stm == Color::WHITE) ? CastlingStatus::White : CastlingStatus::Black;
                for (auto dir : { CastlingStatus::Kingside, CastlingStatus::Queenside }) {
                    const auto right = col & dir;

                    if (pos.CanCastle(occ, us, right)) {
                        list[size++].move = Move(ourKing, pos.CastlingRookSquare(right), FlagCastle);
                    }
                }
            }

            return size;
        }

    }

    template <MoveGenType GenType>
    i32 Generate(const Position& pos, ScoredMove* moveList, i32 size) {
        return pos.Checkers() ? GenAll<GenEvasions>(pos, moveList, 0) :
                                GenAll<GenNonEvasions>(pos, moveList, 0);
    }

    i32 GenerateQS(const Position& pos, ScoredMove* moveList, i32 size) {
        return pos.Checkers() ? GenAll<GenEvasions>(pos, moveList, 0) :
                                GenAll<GenNoisy>(pos, moveList, 0);
    }

    template i32 Generate<PseudoLegal>(const Position&, ScoredMove*, i32);
    template i32 Generate<GenNoisy>(const Position&, ScoredMove*, i32);
    template i32 Generate<GenEvasions>(const Position&, ScoredMove*, i32);
    template i32 Generate<GenNonEvasions>(const Position&, ScoredMove*, i32);

    template<>
    i32 Generate<GenLegal>(const Position& pos, ScoredMove* moveList, i32 size) {
        i32 numMoves = pos.Checkers() ? Generate<GenEvasions>(pos, moveList, 0) :
                                        Generate<GenNonEvasions>(pos, moveList, 0);

        ScoredMove* curr = moveList;
        ScoredMove* end = moveList + numMoves;

        while (curr != end) {
            if (!pos.IsLegal(curr->move)) {
                *curr = *--end;
                numMoves--;
            }
            else {
                ++curr;
            }
        }

        return numMoves;
    }


    template<> i32 GenNormal<HORSIE>(const Position& pos, ScoredMove* list, u64 targets, i32 size) {
        const auto stm = pos.ToMove;
        const auto& bb = pos.bb;
        const auto occ = bb.Occupancy;

        auto ourPieces = bb.Pieces[HORSIE] & bb.Colors[stm];
        ourPieces &= ~pos.BlockingPieces(stm);

        while (ourPieces != 0) {
            auto sq = poplsb(ourPieces);
            auto moves = PseudoAttacks[HORSIE][sq] & targets;
            while (moves != 0) {
                list[size++].move = Move(sq, poplsb(moves));
            }
        }

        return size;
    }

    template<> i32 GenNormal<BISHOP>(const Position& pos, ScoredMove* list, u64 targets, i32 size) {
        const auto stm = pos.ToMove;
        const auto& bb = pos.bb;
        const auto occ = bb.Occupancy;
        const auto ourKingSq = pos.KingSquare(stm);

        auto ourPieces = (bb.Pieces[BISHOP] | bb.Pieces[QUEEN]) & bb.Colors[stm];
        auto unpinned = ourPieces & ~pos.BlockingPieces(stm);
        auto pinned = ourPieces & pos.BlockingPieces(stm);
        const auto pinners = pos.Pinners(Not(stm));

        while (unpinned != 0) {
            auto sq = poplsb(unpinned);
            auto moves = GetBishopMoves(sq, occ) & targets;
            while (moves != 0) {
                list[size++].move = Move(sq, poplsb(moves));
            }
        }

        while (pinned != 0) {
            auto sq = poplsb(pinned);
            auto moves = GetBishopMoves(sq, occ) & targets;
            moves &= RayBB[ourKingSq][sq];
            while (moves != 0) {
                list[size++].move = Move(sq, poplsb(moves));
            }
        }

        return size;
    }

    template<> i32 GenNormal<ROOK>(const Position& pos, ScoredMove* list, u64 targets, i32 size) {
        const auto stm = pos.ToMove;
        const auto& bb = pos.bb;
        const auto occ = bb.Occupancy;
        const auto ourKingSq = pos.KingSquare(stm);

        auto ourPieces = (bb.Pieces[ROOK] | bb.Pieces[QUEEN]) & bb.Colors[stm];
        auto unpinned = ourPieces & ~pos.BlockingPieces(stm);
        auto pinned = ourPieces & pos.BlockingPieces(stm);
        const auto pinners = pos.Pinners(Not(stm));

        while (unpinned != 0) {
            auto sq = poplsb(unpinned);
            auto moves = GetRookMoves(sq, occ) & targets;
            while (moves != 0) {
                list[size++].move = Move(sq, poplsb(moves));
            }
        }

        while (pinned != 0) {
            auto sq = poplsb(pinned);
            auto moves = GetRookMoves(sq, occ) & targets;
            moves &= RayBB[ourKingSq][sq];
            while (moves != 0) {
                list[size++].move = Move(sq, poplsb(moves));
            }
        }

        return size;
    }

}

