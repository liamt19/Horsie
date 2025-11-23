
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

            const auto theirColor = Not(stm);
            const auto north = static_cast<Direction>(ShiftUpDir(stm));
            const auto northwest = (north + Direction::WEST);
            const auto northeast = (north + Direction::EAST);

            const u64 rank7 = (stm == WHITE) ? Rank7BB : Rank2BB;
            const u64 rank3 = (stm == WHITE) ? Rank3BB : Rank6BB;

            const u64 us   = bb.Colors[stm];
            const u64 them = bb.Colors[theirColor];
            const u64 captureSquares = evasions ? pos.Checkers() : them;

            const u64 emptySquares = ~bb.Occupancy;

            const u64 ourPawns = us & bb.Pieces[Piece::PAWN];
            const u64 promos    = ourPawns &  rank7;
            const u64 nonPromos = ourPawns & ~rank7;

            const auto ourKingSq = pos.KingSquare(stm);

            // Pawns that aren't horizontally or diagonally pinned
            const auto pawnsToPush = nonPromos & ~(pos.BlockingPieces(stm) & (RankBB(ourKingSq) | BishopRays[ourKingSq]));

            // Pawns that aren't horizontally or vertically pinned
            const auto pawnsToCap = nonPromos & ~(pos.BlockingPieces(stm) & (RankBB(ourKingSq) | FileBB(ourKingSq)));


            if (!noisyMoves) {
                u64 moves    = Forward(stm, pawnsToPush) & emptySquares;
                u64 twoMoves = Forward(stm, moves & rank3) & emptySquares;

                if (evasions) {
                    //  Only include pushes which block the check
                    moves &= targets;
                    twoMoves &= targets;
                }

                while (moves != 0) {
                    auto sq = poplsb(moves);
                    list[size++].move = Move(sq - north, sq);
                }

                while (twoMoves != 0) {
                    auto sq = poplsb(twoMoves);
                    list[size++].move = Move(sq - north - north, sq);
                }
            }

            if (promos != 0) {
                u64 promoPushes = Shift(north, promos) & emptySquares;
                if (evasions || noisyMoves) {
                    //  Only promote on squares that block the check or capture the checker.
                    promoPushes &= targets;
                }

                while (promoPushes != 0) {
                    auto sq = poplsb(promoPushes);
                    size = MakePromotionChecks<noisyMoves>(list, sq - north, sq, false, size);
                }

                u64 promoCapsW = Shift(northwest, promos) & captureSquares;
                while (promoCapsW != 0) {
                    auto sq = poplsb(promoCapsW);
                    size = MakePromotionChecks<noisyMoves>(list, sq - north - Direction::WEST, sq, true, size);
                }

                u64 promoCapsE = Shift(northeast, promos) & captureSquares;
                while (promoCapsE != 0) {
                    auto sq = poplsb(promoCapsE);
                    size = MakePromotionChecks<noisyMoves>(list, sq - north - Direction::EAST, sq, true, size);
                }
            }

            u64 capsW = Shift(northwest, pawnsToCap) & captureSquares;
            while (capsW != 0) {
                auto sq = poplsb(capsW);
                list[size++].move = Move(sq - north - Direction::WEST, sq);
            }


            u64 capsE = Shift(northeast, pawnsToCap) & captureSquares;
            while (capsE != 0) {
                auto sq = poplsb(capsE);
                list[size++].move = Move(sq - north - Direction::EAST, sq);
            }

            const auto epSq = pos.EPSquare();
            if (epSq != EP_NONE && !noisyMoves) {
                if (evasions && (targets & (SquareBB(epSq + north))) != 0) {
                    //  When in check, we can only en passant if the pawn being captured is the one giving check
                    return size;
                }

                u64 mask = nonPromos & PawnAttackMasks[theirColor][epSq];
                while (mask != 0) {
                    auto sq = poplsb(mask);
                    list[size++].move = Move(sq, epSq, FlagEnPassant);
                }
            }

            return size;
        }

        template <MoveGenType GenType>
        i32 GenAll(const Position& pos, ScoredMove* list) {
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

            i32 size = 0;

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
    i32 Generate(const Position& pos, ScoredMove* moveList) {
        return pos.Checkers() ? GenAll<GenEvasions>(pos, moveList) :
                                GenAll<GenNonEvasions>(pos, moveList);
    }

    i32 GenerateQS(const Position& pos, ScoredMove* moveList) {
        return pos.Checkers() ? GenAll<GenEvasions>(pos, moveList) :
                                GenAll<GenNoisy>(pos, moveList);
    }

    template i32 Generate<PseudoLegal>(const Position&, ScoredMove*);
    template i32 Generate<GenNoisy>(const Position&, ScoredMove*);
    template i32 Generate<GenEvasions>(const Position&, ScoredMove*);
    template i32 Generate<GenNonEvasions>(const Position&, ScoredMove*);

    template<>
    i32 Generate<GenLegal>(const Position& pos, ScoredMove* moveList) {
        i32 numMoves = pos.Checkers() ? Generate<GenEvasions>(pos, moveList) :
                                        Generate<GenNonEvasions>(pos, moveList);

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
        auto   pinned = ourPieces &  pos.BlockingPieces(stm);
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
        auto   pinned = ourPieces &  pos.BlockingPieces(stm);
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

