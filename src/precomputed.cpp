
#include "precomputed.h"

#include "enums.h"
#include "misc.h"
#include "util.h"

#include <cmath>

/*

The magic bitboard code here is taken verbatim from Stockfish (primarily because it is trivial to implement).
Lizard uses the same process for it's magic code:
https://github.com/liamt19/Lizard/blob/d77783b19dcbae0f267d8b30c38258251014780b/Logic/Magic/MagicBitboards.cs#L173

*/

namespace Horsie {

    uint8_t SquareDistance[SQUARE_NB][SQUARE_NB];

    u64 BetweenBB[SQUARE_NB][SQUARE_NB];
    u64 LineBB[SQUARE_NB][SQUARE_NB];
    u64 RayBB[SQUARE_NB][SQUARE_NB];
    u64 XrayBB[SQUARE_NB][SQUARE_NB];
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

        void init_magics(Piece pt, u64 table[], Magic magics[]);

        u64 safe_destination(Square s, i32 step) {
            Square to = Square(static_cast<i32>(s) + step);
            return IsOK(static_cast<i32>(to)) && distance(s, to) <= 2 ? SquareBB(to) : u64(0);
        }
    }

    void Precomputed::init() {
        for (Square s1 = Square::A1; s1 <= Square::H8; ++s1)
            for (Square s2 = Square::A1; s2 <= Square::H8; ++s2)
                SquareDistance[(i32)s1][(i32)s2] = std::max(distance<File>(s1, s2), distance<Rank>(s1, s2));

        init_magics(ROOK, RookTable, RookMagics);
        init_magics(BISHOP, BishopTable, BishopMagics);

        for (Square sq1 = Square::A1; sq1 <= Square::H8; ++sq1) {
            i32 s1 = (i32)sq1;
            PawnAttackMasks[WHITE][s1] = pawn_attacks_bb<WHITE>(SquareBB(sq1));
            PawnAttackMasks[BLACK][s1] = pawn_attacks_bb<BLACK>(SquareBB(sq1));

            for (i32 step : {-9, -8, -7, -1, 1, 7, 8, 9})
                PseudoAttacks[KING][s1] |= safe_destination(sq1, step);

            for (i32 step : {-17, -15, -10, -6, 6, 10, 15, 17})
                PseudoAttacks[HORSIE][s1] |= safe_destination(sq1, step);

            PseudoAttacks[QUEEN][s1] = PseudoAttacks[BISHOP][s1] = attacks_bb<BISHOP>(s1, 0);
            PseudoAttacks[QUEEN][s1] |= PseudoAttacks[ROOK][s1] = attacks_bb<ROOK>(s1, 0);

            RookRays[s1] = PseudoAttacks[ROOK][s1];
            BishopRays[s1] = PseudoAttacks[BISHOP][s1];

            for (i32 s2 = 0; s2 < 64; s2++) {
                if ((RookRays[s1] & SquareBB(s2)) != 0) {
                    BetweenBB[s1][s2] = attacks_bb<ROOK>(s1, SquareBB(s2)) & attacks_bb<ROOK>(s2, SquareBB(s1));
                    LineBB[s1][s2] = BetweenBB[s1][s2] | SquareBB(s2);
                }
                else if ((BishopRays[s1] & SquareBB(s2)) != 0) {
                    BetweenBB[s1][s2] = attacks_bb<BISHOP>(s1, SquareBB(s2)) & attacks_bb<BISHOP>(s2, SquareBB(s1));
                    LineBB[s1][s2] = BetweenBB[s1][s2] | SquareBB(s2);
                }
                else {
                    BetweenBB[s1][s2] = 0;
                    LineBB[s1][s2] = SquareBB(s2);
                }
            }
        }

        for (i32 s1 = 0; s1 < 64; s1++) {
            for (i32 s2 = 0; s2 < 64; s2++) {
                if ((RookRays[s1] & SquareBB(s2)) != 0) {
                    RayBB[s1][s2] = (RookRays[s1] & RookRays[s2]) | SquareBB(s1) | SquareBB(s2);
                    XrayBB[s1][s2] = (attacks_bb(ROOK, s2, SquareBB(s1)) & RookRays[s1]) | SquareBB(s1) | SquareBB(s2);
                }
                else if ((BishopRays[s1] & SquareBB(s2)) != 0) {
                    RayBB[s1][s2] = (BishopRays[s1] & BishopRays[s2]) | SquareBB(s1) | SquareBB(s2);
                    XrayBB[s1][s2] = (attacks_bb(BISHOP, s2, SquareBB(s1)) & BishopRays[s1]) | SquareBB(s1) | SquareBB(s2);
                }
                else {
                    RayBB[s1][s2] = 0;
                    XrayBB[s1][s2] = 0;
                }
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
                while (safe_destination(s, d) && !(occupied & s))
                    attacks |= (s += d);
            }

            return attacks;
        }

        void init_magics(Piece pt, u64 table[], Magic magics[]) {
            // Optimal PRNG seeds to pick the correct magics in the shortest time
            i32 seeds[RANK_NB] = { 728, 10316, 55013, 32803, 12281, 15100, 16645, 255 };

            u64 occupancy[4096], reference[4096], edges, b;
            i32      epoch[4096] = {}, cnt = 0, size = 0;

            for (i32 s = static_cast<i32>(Square::A1); s <= static_cast<i32>(Square::H8); ++s) {
                // Board edges are not considered in the relevant occupancies
                edges = ((Rank1BB | Rank8BB) & ~RankBB(s)) | ((FileABB | FileHBB) & ~FileBB(s));

                // Given a square 's', the mask is the bitboard of sliding attacks from
                // 's' computed on an empty board. The index must be big enough to contain
                // all the attacks for each possible subset of the mask and so is 2 power
                // the number of 1s of the mask. Hence we deduce the size of the shift to
                // apply to the 64 or 32 bits word to get the index.
                Magic& m = magics[s];
                m.mask = sliding_attack(pt, static_cast<Square>(s), 0) & ~edges;
                m.shift = (64 - popcount(m.mask));

                // Set the offset for the attacks table of the square. We have individual
                // table sizes for each square with "Fancy Magic Bitboards".
                m.attacks = s == static_cast<i32>(Square::A1) ? table : magics[s - 1].attacks + size;

                // Use Carry-Rippler trick to enumerate all subsets of masks[s] and
                // store the corresponding sliding attack bitboard in reference[].
                b = size = 0;
                do {
                    occupancy[size] = b;
                    reference[size] = sliding_attack(pt, (Square)s, b);

                    if (HasPext)
                        m.attacks[pext(b, m.mask)] = reference[size];

                    size++;
                    b = (b - m.mask) & m.mask;
                } while (b);

                if (HasPext)
                    continue;

                PRNG rng(seeds[GetIndexRank(s)]);

                // Find a magic for square 's' picking up an (almost) random number
                // until we find the one that passes the verification test.
                for (i32 i = 0; i < size;) {
                    for (m.magic = 0; popcount((m.magic * m.mask) >> 56) < 6;)
                        m.magic = rng.sparse_rand<u64>();

                    // A good magic must map every possible occupancy to an index that
                    // looks up the correct sliding attack in the attacks[s] database.
                    // Note that we build up the database for square 's' as a side
                    // effect of verifying the magic. Keep track of the attempt count
                    // and save it in epoch[], little speed-up trick to avoid resetting
                    // m.attacks[] after every failed attempt.
                    for (++cnt, i = 0; i < size; ++i) {
                        unsigned idx = m.index(occupancy[i]);

                        if (epoch[idx] < cnt) {
                            epoch[idx] = cnt;
                            m.attacks[idx] = reference[i];
                        }
                        else if (m.attacks[idx] != reference[i])
                            break;
                    }
                }
            }
        }
    }


}