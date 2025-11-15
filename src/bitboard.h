#pragma once

#include "defs.h"
#include "enums.h"
#include "types.h"

#include <array>

namespace Horsie {

    class Bitboard {
    public:
        std::array<u64, 6> Pieces = {};
        std::array<u64, 2> Colors = {};
        std::array<i32, 64> PieceTypes = {};
        u64 Occupancy = {};

        Bitboard();
        ~Bitboard() = default;

        constexpr i32 GetColorAtIndex(i32 idx) const { return ((Colors[Color::WHITE] & SquareBB(idx)) != 0) ? Color::WHITE : Color::BLACK; }
        constexpr i32 GetPieceAtIndex(i32 idx) const { return PieceTypes[idx]; }
        constexpr bool IsColorSet(i32 pc, i32 idx) const { return (Colors[pc] & SquareBB(idx)) != 0; }
        constexpr u64 KingMask(i32 pc) const { return Colors[pc] & Pieces[Piece::KING]; }
        constexpr bool Occupied(i32 idx) const { return PieceTypes[idx] != Piece::NONE; }

        void CopyTo(Bitboard& other) const;
        void Reset();
        void AddPiece(i32 idx, i32 pc, i32 pt);
        void RemovePiece(i32 idx, i32 pc, i32 pt);
        void MoveSimple(i32 from, i32 to, i32 pieceColor, i32 pieceType);
        i32 KingIndex(i32 pc) const;
        u64 BlockingPieces(i32 pc, u64* pinners) const;
        u64 AttackersTo(i32 idx, u64 occupied) const;
        u64 AttackMask(i32 idx, i32 pc, i32 pt, u64 occupied) const;

        template<i32 pt>
        u64 AttackMask(i32 idx, i32 pc, u64 occupied) const;

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
