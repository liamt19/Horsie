#pragma once

#ifndef PRECOMPUTED_H
#define PRECOMPUTED_H

#include "types.h"

namespace Horsie {
    extern uint8_t SquareDistance[SQUARE_NB][SQUARE_NB];
    extern ulong BetweenBB[SQUARE_NB][SQUARE_NB];
    extern ulong LineBB[SQUARE_NB][SQUARE_NB];
    extern ulong RayBB[SQUARE_NB][SQUARE_NB];
    extern ulong XrayBB[SQUARE_NB][SQUARE_NB];
    extern ulong PseudoAttacks[PIECE_NB][SQUARE_NB];
    extern ulong PawnAttackMasks[COLOR_NB][SQUARE_NB];
    extern ulong HorsieMasks[SQUARE_NB];
    extern ulong RookRays[SQUARE_NB];
    extern ulong BishopRays[SQUARE_NB];

    extern int LogarithmicReductionTable[MaxPly][MoveListSize];
    extern int LMPTable[2][MaxDepth];

    namespace Precomputed {
        void init();
    }

    struct Magic {
        ulong mask;
        ulong magic;
        ulong* attacks;
        unsigned shift;

        // Compute the attack's index using the 'magic bitboards' approach
        unsigned index(ulong occupied) const {

            if (HasPext)
                return unsigned(pext(occupied, mask));

            return unsigned(((occupied & mask) * magic) >> shift);
        }
    };

    extern Magic RookMagics[SQUARE_NB];
    extern Magic BishopMagics[SQUARE_NB];


    template<Color C>
    constexpr ulong pawn_attacks_bb(ulong b) {
        return C == WHITE ? Shift<NORTH_WEST>(b) | Shift<NORTH_EAST>(b)
            : Shift<SOUTH_WEST>(b) | Shift<SOUTH_EAST>(b);
    }

    inline ulong pawn_attacks_bb(Color c, int s) { return PawnAttackMasks[c][s]; }
    inline ulong pawn_attacks_bb(Color c, Square s) { return pawn_attacks_bb(c, (int)s); }

    inline ulong line_bb(int s1, int s2) { return LineBB[(int)s1][(int)s2]; }
    inline ulong line_bb(Square s1, Square s2) { return line_bb((int)s1, (int)s2); }

    inline ulong between_bb(int s1, int s2) { return BetweenBB[s1][s2]; }
    inline ulong between_bb(Square s1, Square s2) { return between_bb((int)s1, (int)s2); }

    inline bool aligned(Square s1, Square s2, Square s3) { return line_bb(s1, s2) & s3; }

    template<typename T1 = Square>
    inline int distance(Square x, Square y);

    template<>
    inline int distance<File>(Square x, Square y) {
        return std::abs(GetIndexFile((int)x) - GetIndexFile((int)y));
    }

    template<>
    inline int distance<Rank>(Square x, Square y) {
        return std::abs(GetIndexRank((int)x) - GetIndexRank((int)y));
    }

    template<>
    inline int distance<Square>(Square x, Square y) { return SquareDistance[(int)x][(int)y]; }


    template<typename T1 = int>
    inline int distance(int x, int y);

    template<>
    inline int distance<File>(int x, int y) {
        return std::abs(GetIndexFile((int)x) - GetIndexFile((int)y));
    }

    template<>
    inline int distance<Rank>(int x, int y) {
        return std::abs(GetIndexRank((int)x) - GetIndexRank((int)y));
    }

    template<>
    inline int distance<int>(int x, int y) { return SquareDistance[(int)x][(int)y]; }





    inline int edge_distance(File f) { return std::min(f, File(FILE_H - f)); }

    // Returns the pseudo attacks of the given piece type
    // assuming an empty board.
    template<Piece Pt>
    inline ulong attacks_bb(int s) {
        assert((Pt != PAWN) && (IsOK(s)));
        return PseudoAttacks[Pt][(int)s];
    }


    // Returns the attacks by the given piece
    // assuming the board is occupied according to the passed Bitboard.
    // Sliding piece attacks do not continue passed an occupied square.
    template<Piece Pt>
    inline ulong attacks_bb(int s, ulong occupied) {
        switch (Pt)
        {
        case BISHOP:
            return BishopMagics[(int)s].attacks[BishopMagics[(int)s].index(occupied)];
        case ROOK:
            return RookMagics[(int)s].attacks[RookMagics[(int)s].index(occupied)];
        case QUEEN:
            return attacks_bb<BISHOP>(s, occupied) | attacks_bb<ROOK>(s, occupied);
        default:
            return PseudoAttacks[Pt][(int)s];
        }
    }

    // Returns the attacks by the given piece
    // assuming the board is occupied according to the passed Bitboard.
    // Sliding piece attacks do not continue passed an occupied square.
    inline ulong attacks_bb(int pt, int s, ulong occupied) {
        switch (pt)
        {
        case BISHOP:
            return attacks_bb<BISHOP>(s, occupied);
        case ROOK:
            return attacks_bb<ROOK>(s, occupied);
        case QUEEN:
            return attacks_bb<BISHOP>(s, occupied) | attacks_bb<ROOK>(s, occupied);
        default:
            return PseudoAttacks[pt][(int)s];
        }
    }
}


#endif // !PRECOMPUTED_H