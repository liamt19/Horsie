
#include "movegen.h"

#include "bitboard.h"
#include "enums.h"
#include "position.h"
#include "precomputed.h"
#include "types.h"


namespace Horsie {

    namespace {

        template <bool noisyMoves>
        void MakePromotionChecks(MoveList& list, Square from, Square promotionSquare, bool isCapture) {
            list.Add(Move::Promotion(from, promotionSquare, Piece::QUEEN));

            if (!noisyMoves || isCapture) {
                list.Add(Move::Promotion(from, promotionSquare, Piece::HORSIE));
                list.Add(Move::Promotion(from, promotionSquare, Piece::ROOK));
                list.Add(Move::Promotion(from, promotionSquare, Piece::BISHOP));
            }
        }


        template <MoveGenType GenType>
        void GenPawns(const Position& pos, MoveList& list, u64 targets) {
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
                    auto to = poplsb(moves);
                    list.Add(Move(to - up, to));
                }

                while (twoMoves != 0) {
                    auto to = poplsb(twoMoves);
                    list.Add(Move(to - up - up, to));
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
                    auto to = poplsb(promotions);
                    MakePromotionChecks<noisyMoves>(list, to - up, to, false);
                }

                while (promotionCapturesL != 0) {
                    auto to = poplsb(promotionCapturesL);
                    MakePromotionChecks<noisyMoves>(list, to - up - Direction::WEST, to, true);
                }

                while (promotionCapturesR != 0) {
                    auto to = poplsb(promotionCapturesR);
                    MakePromotionChecks<noisyMoves>(list, to - up - Direction::EAST, to, true);
                }
            }

            u64 capturesL = Shift(UpLeft, notPromotingPawns) & captureSquares;
            u64 capturesR = Shift(UpRight, notPromotingPawns) & captureSquares;

            while (capturesL != 0) {
                auto to = poplsb(capturesL);
                list.Add(Move(to - up - Direction::WEST, to));
            }

            while (capturesR != 0) {
                auto to = poplsb(capturesR);
                list.Add(Move(to - up - Direction::EAST, to));
            }

            const auto epSq = pos.EPSquare();
            if (epSq != EP_NONE && !noisyMoves) {
                if (evasions && (targets & (SquareBB(epSq + up))) != 0) {
                    //  When in check, we can only en passant if the pawn being captured is the one giving check
                    return;
                }

                u64 mask = notPromotingPawns & PawnAttackMasks[theirColor][epSq];
                while (mask != 0) {
                    auto from = poplsb(mask);
                    list.Add(Move::EnPassant(from, epSq));
                }
            }

        }

        void GenNormal(const Position& pos, MoveList& list, Piece pt, u64 targets) {
            const Color stm = pos.ToMove;
            const Bitboard& bb = pos.bb;
            const u64 occ = bb.Occupancy;
            u64 ourPieces = bb.Pieces[pt] & bb.Colors[stm];

            while (ourPieces != 0) {
                auto idx = poplsb(ourPieces);
                u64 moves = bb.AttackMask(idx, stm, pt, occ) & targets;

                while (moves != 0) {
                    auto to = poplsb(moves);
                    list.Add(Move(idx, to));
                }
            }
        }

        template <MoveGenType GenType>
        void GenAll(const Position& pos, MoveList& list) {
            constexpr bool noisyMoves = GenType == GenNoisy;
            constexpr bool evasions = GenType == GenEvasions;
            constexpr bool nonEvasions = GenType == GenNonEvasions;

            const Color stm = pos.ToMove;
            const Bitboard& bb = pos.bb;

            u64 us   = bb.Colors[stm];
            u64 them = bb.Colors[Not(stm)];
            u64 occ  = bb.Occupancy;

            auto ourKing = pos.KingSquare(stm);
            u64 targets = 0;

            // If we are generating evasions and in double check, then skip non-king moves.
            if (!(evasions && MoreThanOne(pos.Checkers()))) {
                targets = evasions    ?  LineBB[ourKing][lsb(pos.Checkers())]
                        : nonEvasions ? ~us
                        : noisyMoves  ?  them
                        :               ~occ;

                GenPawns<GenType>(pos, list, targets);
                GenNormal(pos, list, HORSIE, targets);
                GenNormal(pos, list, BISHOP, targets);
                GenNormal(pos, list, ROOK, targets);
                GenNormal(pos, list, QUEEN, targets);
            }

            u64 moves = PseudoAttacks[KING][ourKing] & (evasions ? ~us : targets);
            while (moves != 0) {
                auto to = poplsb(moves);
                list.Add(Move(ourKing, to));
            }

            if (nonEvasions) {
                const auto stmCr = (stm == Color::WHITE) ? CastlingStatus::White : CastlingStatus::Black;
                if (pos.HasCastlingRight(stmCr)) {
                    if (pos.CanCastle(occ, us, stmCr & CastlingStatus::Kingside))
                        list.Add(Move::Castle(ourKing, pos.CastlingRookSquare(stmCr & CastlingStatus::Kingside)));

                    if (pos.CanCastle(occ, us, stmCr & CastlingStatus::Queenside))
                        list.Add(Move::Castle(ourKing, pos.CastlingRookSquare(stmCr & CastlingStatus::Queenside)));
                }
            }

        }

    }

    template <MoveGenType GenType>
    void Generate(const Position& pos, MoveList& list) {
        pos.Checkers() ? GenAll<GenEvasions>(pos, list) : GenAll<GenNonEvasions>(pos, list);
    }

    void GenerateQS(const Position& pos, MoveList& list) {
        pos.Checkers() ? GenAll<GenEvasions>(pos, list) : GenAll<GenNoisy>(pos, list);
    }

    template void Generate<PseudoLegal>(const Position&, MoveList&);
    template void Generate<GenNoisy>(const Position&, MoveList&);
    template void Generate<GenEvasions>(const Position&, MoveList&);
    template void Generate<GenNonEvasions>(const Position&, MoveList&);

    template<>
    void Generate<GenLegal>(const Position& pos, MoveList& list) {
        list.Clear();
        Generate<PseudoLegal>(pos, list);

        const auto stm = pos.ToMove;
        const auto ourKing   = pos.KingSquare(stm);
        const auto theirKing = pos.KingSquare(Not(stm));
        u64 pinned = pos.BlockingPieces(stm);

        u32 curr = 0;
        u32 end = list.Size();

        while (curr != end) {
            if (!pos.IsLegal(list[curr].move, ourKing, theirKing, pinned)) {
                end--;
                list.Swap(curr, end);
            }
            else {
                curr++;
            }
        }

        list.Resize(curr);
    }
}

