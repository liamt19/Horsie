
#include "precomputed.h"
#include "misc.h"
#include "util.h"
#include "enums.h"

#include <cmath>

constexpr bool Is64Bit = true;

namespace Horsie {

    uint8_t SquareDistance[SQUARE_NB][SQUARE_NB];

    ulong BetweenBB[SQUARE_NB][SQUARE_NB];
    ulong LineBB[SQUARE_NB][SQUARE_NB];
    ulong RayBB[SQUARE_NB][SQUARE_NB];
    ulong XrayBB[SQUARE_NB][SQUARE_NB];
    ulong PseudoAttacks[PIECE_NB][SQUARE_NB];

    ulong PawnAttackMasks[COLOR_NB][SQUARE_NB];
    ulong RookRays[SQUARE_NB];
    ulong BishopRays[SQUARE_NB];

    Magic RookMagics[SQUARE_NB];
    Magic BishopMagics[SQUARE_NB];

    int LogarithmicReductionTable[MaxPly][MoveListSize];
    int LMPTable[2][MaxDepth];

    namespace {
        ulong RookTable[0x19000];   // To store rook attacks
        ulong BishopTable[0x1480];  // To store bishop attacks

        void init_magics(Piece pt, ulong table[], Magic magics[]);

        ulong safe_destination(Square s, int step) {
            Square to = Square(s + step);
            return IsOK((int)to) && distance(s, to) <= 2 ? SquareBB(to) : ulong(0);
        }
    }

    void Precomputed::init() {

        for (Square s1 = Square::A1; s1 <= Square::H8; ++s1)
            for (Square s2 = Square::A1; s2 <= Square::H8; ++s2)
                SquareDistance[(int)s1][(int)s2] = std::max(distance<File>(s1, s2), distance<Rank>(s1, s2));

        init_magics(ROOK, RookTable, RookMagics);
        init_magics(BISHOP, BishopTable, BishopMagics);

        for (Square sq1 = Square::A1; sq1 <= Square::H8; ++sq1)
        {
            int s1 = (int)sq1;
            PawnAttackMasks[WHITE][s1] = pawn_attacks_bb<WHITE>(SquareBB(sq1));
            PawnAttackMasks[BLACK][s1] = pawn_attacks_bb<BLACK>(SquareBB(sq1));

            for (int step : {-9, -8, -7, -1, 1, 7, 8, 9})
                PseudoAttacks[KING][s1] |= safe_destination(sq1, step);

            for (int step : {-17, -15, -10, -6, 6, 10, 15, 17})
                PseudoAttacks[HORSIE][s1] |= safe_destination(sq1, step);

            PseudoAttacks[QUEEN][s1] = PseudoAttacks[BISHOP][s1] = attacks_bb<BISHOP>(s1, 0);
            PseudoAttacks[QUEEN][s1] |= PseudoAttacks[ROOK][s1] = attacks_bb<ROOK>(s1, 0);

            RookRays[s1] = PseudoAttacks[ROOK][s1];
            BishopRays[s1] = PseudoAttacks[BISHOP][s1];
            
            for (int s2 = 0; s2 < 64; s2++)
            {
                if ((RookRays[s1] & SquareBB(s2)) != 0)
                {
                    BetweenBB[s1][s2] = attacks_bb<ROOK>(s1, SquareBB(s2)) & attacks_bb<ROOK>(s2, SquareBB(s1));
                    LineBB[s1][s2] = BetweenBB[s1][s2] | SquareBB(s2);
                }
                else if ((BishopRays[s1] & SquareBB(s2)) != 0)
                {
                    BetweenBB[s1][s2] = attacks_bb<BISHOP>(s1, SquareBB(s2)) & attacks_bb<BISHOP>(s2, SquareBB(s1));
                    LineBB[s1][s2] = BetweenBB[s1][s2] | SquareBB(s2);
                }
                else
                {
                    BetweenBB[s1][s2] = 0;
                    LineBB[s1][s2] = SquareBB(s2);
                }
            }            
        }

        for (int s1 = 0; s1 < 64; s1++) {
            for (int s2 = 0; s2 < 64; s2++)
            {
                if ((RookRays[s1] & SquareBB(s2)) != 0)
                {
                    RayBB[s1][s2] = (RookRays[s1] & RookRays[s2]) | SquareBB(s1) | SquareBB(s2);
                    XrayBB[s1][s2] = (attacks_bb(ROOK, s2, SquareBB(s1)) & RookRays[s1]) | SquareBB(s1) | SquareBB(s2);
                }
                else if ((BishopRays[s1] & SquareBB(s2)) != 0)
                {
                    RayBB[s1][s2] = (BishopRays[s1] & BishopRays[s2]) | SquareBB(s1) | SquareBB(s2);
                    XrayBB[s1][s2] = (attacks_bb(BISHOP, s2, SquareBB(s1)) & BishopRays[s1]) | SquareBB(s1) | SquareBB(s2);
                }
                else
                {
                    RayBB[s1][s2] = 0;
                    XrayBB[s1][s2] = 0;
                }
            }
        }


        for (int depth = 0; depth < MaxPly; depth++)
        {
            for (int moveIndex = 0; moveIndex < MoveListSize; moveIndex++)
            {
                if (depth == 0 || moveIndex == 0) {
                    LogarithmicReductionTable[depth][moveIndex] = 0;
                    continue;
                }

                LogarithmicReductionTable[depth][moveIndex] = int((std::log(depth) * std::log(moveIndex) / 2.25) + 0.25);

                if (LogarithmicReductionTable[depth][moveIndex] < 1)
                {
                    LogarithmicReductionTable[depth][moveIndex] = 0;
                }
            }
        }

        const int not_improving = 0;
        const int improving = 1;
        for (int depth = 0; depth < MaxDepth; depth++)
        {
            LMPTable[not_improving][depth] = (3 + (depth * depth)) / 2;
            LMPTable[    improving][depth] =  3 + (depth * depth);
        }
    }


    namespace {
        ulong sliding_attack(Piece pt, Square sq, ulong occupied) {

            ulong  attacks = 0;
            Direction RookDirections[4] = { NORTH, SOUTH, EAST, WEST };
            Direction BishopDirections[4] = { NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST };

            for (Direction d : (pt == ROOK ? RookDirections : BishopDirections))
            {
                Square s = sq;
                while (safe_destination(s, d) && !(occupied & s))
                    attacks |= (s += d);
            }

            return attacks;
        }

        void init_magics(Piece pt, ulong table[], Magic magics[]) {

            // Optimal PRNG seeds to pick the correct magics in the shortest time
            int seeds[RANK_NB] = { 728, 10316, 55013, 32803, 12281, 15100, 16645, 255 };

            ulong occupancy[4096], reference[4096], edges, b;
            int      epoch[4096] = {}, cnt = 0, size = 0;

            for (int s = (int)Square::A1; s <= (int)Square::H8; ++s)
            {
                // Board edges are not considered in the relevant occupancies
                edges = ((Rank1BB | Rank8BB) & ~RankBB(s)) | ((FileABB | FileHBB) & ~FileBB(s));

                // Given a square 's', the mask is the bitboard of sliding attacks from
                // 's' computed on an empty board. The index must be big enough to contain
                // all the attacks for each possible subset of the mask and so is 2 power
                // the number of 1s of the mask. Hence we deduce the size of the shift to
                // apply to the 64 or 32 bits word to get the index.
                Magic& m = magics[s];
                m.mask = sliding_attack(pt, (Square)s, 0) & ~edges;
                m.shift = (64 - popcount(m.mask));

                // Set the offset for the attacks table of the square. We have individual
                // table sizes for each square with "Fancy Magic Bitboards".
                m.attacks = s == (int)Square::A1 ? table : magics[s - 1].attacks + size;

                // Use Carry-Rippler trick to enumerate all subsets of masks[s] and
                // store the corresponding sliding attack bitboard in reference[].
                b = size = 0;
                do
                {
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
                for (int i = 0; i < size;)
                {
                    for (m.magic = 0; popcount((m.magic * m.mask) >> 56) < 6;)
                        m.magic = rng.sparse_rand<ulong>();

                    // A good magic must map every possible occupancy to an index that
                    // looks up the correct sliding attack in the attacks[s] database.
                    // Note that we build up the database for square 's' as a side
                    // effect of verifying the magic. Keep track of the attempt count
                    // and save it in epoch[], little speed-up trick to avoid resetting
                    // m.attacks[] after every failed attempt.
                    for (++cnt, i = 0; i < size; ++i)
                    {
                        unsigned idx = m.index(occupancy[i]);

                        if (epoch[idx] < cnt)
                        {
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