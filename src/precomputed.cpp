
#include "precomputed.h"

#include "enums.h"
#include "util.h"

#include <cmath>

/*
The magic bitboard code here is adapted from Stockfish (primarily because it is trivial to implement).
Lizard uses the same process for it's magic code:
https://github.com/liamt19/Lizard/blob/d77783b19dcbae0f267d8b30c38258251014780b/Logic/Magic/MagicBitboards.cs#L173
*/

namespace Horsie {

    u64 BetweenBB[SQUARE_NB][SQUARE_NB];
    u64 LineBB[SQUARE_NB][SQUARE_NB];
    u64 RayBB[SQUARE_NB][SQUARE_NB];
    u64 PseudoAttacks[PIECE_NB][SQUARE_NB];

    u64 PawnAttackMasks[COLOR_NB][SQUARE_NB];
    u64 RookRays[SQUARE_NB];
    u64 BishopRays[SQUARE_NB];

    Magic RookMagics[SQUARE_NB];
    Magic BishopMagics[SQUARE_NB];

    i32 LogarithmicReductionTable[MaxPly][MoveListSize];
    i32 LMPTable[2][MaxDepth];

    namespace {
        u64 RookTable[0x19000];   // To store rook attacks
        u64 BishopTable[0x1480];  // To store bishop attacks

        void InitPextMagics(Piece pt, u64 table[], Magic magics[]);
        void InitNormalMagics(Piece pt, u64 table[], Magic magics[]);

        constexpr bool DirectionOK(Square sq, Direction dir) {
            if (sq + dir < Square::A1 || sq + dir > Square::H8) {
                return false;
            }

            i32 rankDistance = std::abs(GetIndexRank(sq) - GetIndexRank(sq + dir));
            i32 fileDistance = std::abs(GetIndexFile(sq) - GetIndexFile(sq + dir));
            return std::max(rankDistance, fileDistance) <= 2;
        }
    }

    void Precomputed::Init() {

        if (UsePext) {
            InitPextMagics(ROOK, RookTable, RookMagics);
            InitPextMagics(BISHOP, BishopTable, BishopMagics);
        }
        else {
            InitNormalMagics(ROOK, RookTable, RookMagics);
            InitNormalMagics(BISHOP, BishopTable, BishopMagics);
        }

        for (Square s1 = Square::A1; s1 <= Square::H8; ++s1) {
            PawnAttackMasks[WHITE][s1] = Shift<NORTH_WEST>(SquareBB(s1)) | Shift<NORTH_EAST>(SquareBB(s1));
            PawnAttackMasks[BLACK][s1] = Shift<SOUTH_WEST>(SquareBB(s1)) | Shift<SOUTH_EAST>(SquareBB(s1));

            for (i32 step : {7, 8, 9, -1, 1, -7, -8, -9}) {
                if (DirectionOK(s1, static_cast<Direction>(step)))
                    PseudoAttacks[KING][s1] |= (1ULL << (static_cast<i32>(s1) + step));
            }

            for (i32 step : {6, 10, 15, 17, -6, -10, -15, -17}) {
                if (DirectionOK(s1, static_cast<Direction>(step)))
                    PseudoAttacks[HORSIE][s1] |= (1ULL << (static_cast<i32>(s1) + step));
            }

            PseudoAttacks[BISHOP][s1] = BishopRays[s1] = attacks_bb<BISHOP>(s1, 0);
            PseudoAttacks[ROOK][s1] = RookRays[s1] = attacks_bb<ROOK>(s1, 0);
            PseudoAttacks[QUEEN][s1] = PseudoAttacks[BISHOP][s1] | PseudoAttacks[ROOK][s1];

            for (Square s2 = Square::A1; s2 <= Square::H8; ++s2) {
                if ((RookRays[s1] & SquareBB(s2)) != 0) {
                    BetweenBB[s1][s2] = attacks_bb<ROOK>(s1, SquareBB(s2)) & attacks_bb<ROOK>(s2, SquareBB(s1));
                    RayBB[s1][s2] = (attacks_bb<ROOK>(s1, 0) & attacks_bb<ROOK>(s2, 0)) | SquareBB(s1) | SquareBB(s2);
                }
                else if ((BishopRays[s1] & SquareBB(s2)) != 0) {
                    BetweenBB[s1][s2] = attacks_bb<BISHOP>(s1, SquareBB(s2)) & attacks_bb<BISHOP>(s2, SquareBB(s1));
                    RayBB[s1][s2] = (attacks_bb<BISHOP>(s1, 0) & attacks_bb<BISHOP>(s2, 0)) | SquareBB(s1) | SquareBB(s2);
                }
                else {
                    BetweenBB[s1][s2] = 0;
                    RayBB[s1][s2] = 0;
                }

                LineBB[s1][s2] = BetweenBB[s1][s2] | SquareBB(s2);
            }
        }

        for (i32 depth = 0; depth < MaxPly; depth++) {
            for (i32 moveIndex = 0; moveIndex < MoveListSize; moveIndex++) {
                if (depth == 0 || moveIndex == 0) {
                    LogarithmicReductionTable[depth][moveIndex] = 0;
                    continue;
                }

                LogarithmicReductionTable[depth][moveIndex] = i32((std::log(depth) * std::log(moveIndex) / 2.25) + 0.25);

                if (LogarithmicReductionTable[depth][moveIndex] < 1) {
                    LogarithmicReductionTable[depth][moveIndex] = 0;
                }
            }
        }

        for (i32 depth = 0; depth < MaxDepth; depth++) {
            LMPTable[0][depth] = (3 + (depth * depth)) / 2;
            LMPTable[1][depth] = 3 + (depth * depth);
        }
    }

    namespace {
        u64 sliding_attack(Piece pt, Square sq, u64 occupied) {
            u64  attacks = 0;
            Direction RookDirections[4] = { NORTH, SOUTH, EAST, WEST };
            Direction BishopDirections[4] = { NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST };

            for (Direction d : (pt == ROOK ? RookDirections : BishopDirections)) {
                Square s = sq;
                while (DirectionOK(s, d) && !(occupied & s))
                    attacks |= (s += d);
            }

            return attacks;
        }

        void InitPextMagics(Piece pt, u64 table[], Magic magics[]) {
            u64 combo = 0;
            u64 iter = 0;
            u64 edges = 0;
            for (Square sq = Square::A1; sq <= Square::H8; ++sq) {
                auto& m = magics[sq];
                
                edges = ((Rank1BB | Rank8BB) & ~RankBB(sq)) | ((FileABB | FileHBB) & ~FileBB(sq));
                m.mask = sliding_attack(pt, static_cast<Square>(sq), 0) & ~edges;
                m.shift = static_cast<i32>(64 - popcount(m.mask));
                if (sq == 0) {
                    m.attacks = (u64*)(table + 0);
                }
                else {
                    m.attacks = magics[sq - 1].attacks + iter;
                }

                combo = 0;
                iter = 0;
                do {
                    m.attacks[pext(combo, m.mask)] = sliding_attack(pt, static_cast<Square>(sq), combo);
                    iter++;
                    combo = (combo - m.mask) & m.mask;
                } while (combo != 0);
            }
        }

        void InitNormalMagics(Piece pt, u64 table[], Magic magics[]) {
            u64 offset = 0;

            for (Square sq = Square::A1; sq <= Square::H8; ++sq) {
                auto& m = magics[sq];

                m.number = (pt == BISHOP) ? BishopMagicNumbers[sq] : RookMagicNumbers[sq];
                m.mask = sliding_attack(pt, sq, 0) & ~(((Rank1BB | Rank8BB) & ~RankBB(sq)) | ((FileABB | FileHBB) & ~FileBB(sq)));
                m.shift = static_cast<i32>(64 - popcount(m.mask));
                m.attacks = &table[offset];

                u64 combo = 0;
                u32 iter = 0;
                do {
                    m.attacks[(combo * m.number) >> m.shift] = sliding_attack(pt, sq, combo);
                    iter++;
                    combo = (combo - m.mask) & m.mask;
                } while (combo != 0);

                offset += iter;
            }
        }
    }
}
