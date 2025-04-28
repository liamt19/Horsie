#pragma once

#include "defs.h"
#include "enums.h"
#include "types.h"

namespace Horsie {

    class Bitboard {
    public:
        u64 Pieces[6];
        u64 Colors[2];
        Piece PieceTypes[64];

        u64 Occupancy = 0;

        Bitboard();
        ~Bitboard() = default;

        constexpr Color GetColorAtIndex(Square idx) const { return ((Colors[Color::WHITE] & SquareBB(idx)) != 0) ? Color::WHITE : Color::BLACK; }
        constexpr Piece GetPieceAtIndex(Square idx) const { return PieceTypes[idx]; }
        constexpr u64 KingMask(Color pc) const { return Colors[pc] & Pieces[Piece::KING]; }
        constexpr bool Occupied(Square idx) const { return PieceTypes[idx] != Piece::NONE; }

        void CopyTo(Bitboard& other) const;
        void Reset();
        void AddPiece(Square idx, Color pc, Piece pt);
        void RemovePiece(Square idx, Color pc, Piece pt);
        void MoveSimple(Square from, Square to, Color pc, Piece pt);
        Square KingIndex(Color pc) const;
        u64 BlockingPieces(Color pc, u64* pinners) const;
        u64 AttackersTo(Square idx, u64 occupied) const;
        u64 AttackMask(Square idx, Color pc, Piece pt, u64 occupied) const;

        template<Piece pt>
        u64 AttackMask(Square idx, Color pc, u64 occupied) const;

        template<Piece pt>
        inline u64 ThreatsBy(Color pc) const {
            u64 mask{};
            auto pieces = Pieces[pt] & Colors[pc];
            while (pieces != 0)
                mask |= AttackMask<pt>(poplsb(pieces), pc, Occupancy);

            return mask;
        }
    };
}
