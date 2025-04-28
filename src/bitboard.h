#pragma once

#include "defs.h"
#include "enums.h"
#include "types.h"

namespace Horsie {

    class Bitboard {
    public:
        u64 Pieces[6];
        u64 Colors[2];
        i32 PieceTypes[64];

        u64 Occupancy = 0;

        Bitboard();
        ~Bitboard() = default;

        constexpr i32 GetColorAtIndex(Square idx) const { return ((Colors[Color::WHITE] & SquareBB(idx)) != 0) ? Color::WHITE : Color::BLACK; }
        constexpr i32 GetPieceAtIndex(Square idx) const { return PieceTypes[idx]; }
        constexpr u64 KingMask(i32 pc) const { return Colors[pc] & Pieces[Piece::KING]; }
        constexpr bool Occupied(Square idx) const { return PieceTypes[idx] != Piece::NONE; }

        void CopyTo(Bitboard& other) const;
        void Reset();
        void AddPiece(Square idx, i32 pc, i32 pt);
        void RemovePiece(Square idx, i32 pc, i32 pt);
        void MoveSimple(Square from, Square to, i32 pc, i32 pt);
        Square KingIndex(i32 pc) const;
        u64 BlockingPieces(i32 pc, u64* pinners) const;
        u64 AttackersTo(Square idx, u64 occupied) const;
        u64 AttackMask(Square idx, i32 pc, i32 pt, u64 occupied) const;

        template<i32 pt>
        u64 AttackMask(Square idx, i32 pc, u64 occupied) const;

        template<i32 pt>
        inline u64 ThreatsBy(i32 pc) const {
            u64 mask{};
            auto pieces = Pieces[pt] & Colors[pc];
            while (pieces != 0)
                mask |= AttackMask<pt>(poplsb(pieces), pc, Occupancy);

            return mask;
        }
    };
}
