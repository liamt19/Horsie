#pragma once

#include "defs.h"
#include "enums.h"
#include "types.h"
#include "util.h"

namespace Horsie {
    extern u8 SquareDistance[SQUARE_NB][SQUARE_NB];
    extern u64 BetweenBB[SQUARE_NB][SQUARE_NB];
    extern u64 LineBB[SQUARE_NB][SQUARE_NB];
    extern u64 RayBB[SQUARE_NB][SQUARE_NB];
    extern u64 XrayBB[SQUARE_NB][SQUARE_NB];
    extern u64 PseudoAttacks[PIECE_NB][SQUARE_NB];
    extern u64 PawnAttackMasks[COLOR_NB][SQUARE_NB];
    extern u64 HorsieMasks[SQUARE_NB];
    extern u64 RookRays[SQUARE_NB];
    extern u64 BishopRays[SQUARE_NB];

    extern i32 LogarithmicReductionTable[MaxPly][MoveListSize];
    extern i32 LMPTable[2][MaxDepth];

    namespace Precomputed {
        void init();
    }

    struct Magic {
        u64 mask;
        u64 magic;
        u64* attacks;
        unsigned shift;

        // Compute the attack's index using the 'magic bitboards' approach
        unsigned index(u64 occupied) const {

            if (HasPext)
                return unsigned(pext(occupied, mask));

            return unsigned(((occupied & mask) * magic) >> shift);
        }
    };

    extern Magic RookMagics[SQUARE_NB];
    extern Magic BishopMagics[SQUARE_NB];


    template<Color C>
    constexpr u64 pawn_attacks_bb(u64 b) {
        return C == WHITE ? Shift<NORTH_WEST>(b) | Shift<NORTH_EAST>(b)
                          : Shift<SOUTH_WEST>(b) | Shift<SOUTH_EAST>(b);
    }

    inline u64 pawn_attacks_bb(Color c, i32 s) { return PawnAttackMasks[c][s]; }
    inline u64 pawn_attacks_bb(Color c, Square s) { return pawn_attacks_bb(c, (i32)s); }

    inline u64 line_bb(i32 s1, i32 s2) { return LineBB[(i32)s1][(i32)s2]; }
    inline u64 line_bb(Square s1, Square s2) { return line_bb((i32)s1, (i32)s2); }

    inline u64 between_bb(i32 s1, i32 s2) { return BetweenBB[s1][s2]; }
    inline u64 between_bb(Square s1, Square s2) { return between_bb((i32)s1, (i32)s2); }

    inline bool aligned(Square s1, Square s2, Square s3) { return line_bb(s1, s2) & s3; }

    template<typename T1 = Square>
    inline i32 distance(Square x, Square y);

    template<>
    inline i32 distance<File>(Square x, Square y) {
        return std::abs(GetIndexFile((i32)x) - GetIndexFile((i32)y));
    }

    template<>
    inline i32 distance<Rank>(Square x, Square y) {
        return std::abs(GetIndexRank((i32)x) - GetIndexRank((i32)y));
    }

    template<>
    inline i32 distance<Square>(Square x, Square y) { return SquareDistance[(i32)x][(i32)y]; }


    template<typename T1 = i32>
    inline i32 distance(i32 x, i32 y);

    template<>
    inline i32 distance<File>(i32 x, i32 y) {
        return std::abs(GetIndexFile((i32)x) - GetIndexFile((i32)y));
    }

    template<>
    inline i32 distance<Rank>(i32 x, i32 y) {
        return std::abs(GetIndexRank((i32)x) - GetIndexRank((i32)y));
    }

    template<>
    inline i32 distance<i32>(i32 x, i32 y) { return SquareDistance[(i32)x][(i32)y]; }

    inline i32 edge_distance(File f) { return std::min(f, File(FILE_H - f)); }

    template<Piece Pt>
    inline u64 attacks_bb(i32 s) {
        return PseudoAttacks[Pt][(i32)s];
    }

    template<Piece Pt>
    inline u64 attacks_bb(i32 s, u64 occupied) {
        switch (Pt) {
        case BISHOP: return BishopMagics[(i32)s].attacks[BishopMagics[(i32)s].index(occupied)];
        case ROOK: return RookMagics[(i32)s].attacks[RookMagics[(i32)s].index(occupied)];
        case QUEEN: return attacks_bb<BISHOP>(s, occupied) | attacks_bb<ROOK>(s, occupied);
        default: return PseudoAttacks[Pt][(i32)s];
        }
    }

    inline u64 attacks_bb(i32 pt, i32 s, u64 occupied) {
        switch (pt) {
        case BISHOP: return attacks_bb<BISHOP>(s, occupied);
        case ROOK: return attacks_bb<ROOK>(s, occupied);
        case QUEEN: return attacks_bb<BISHOP>(s, occupied) | attacks_bb<ROOK>(s, occupied);
        default: return PseudoAttacks[pt][(i32)s];
        }
    }
}
