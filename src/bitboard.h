#pragma once

#ifndef BITBOARD_H
#define BITBOARD_H

#include <string>
#include <cassert>

#include "nnue/accumulator.h"
#include "util.h"

namespace Horsie {

    class Bitboard
    {
    public:
        u64 Pieces[6];
        u64 Colors[2];
        i32 PieceTypes[64];

        u64 Occupancy = 0;

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
    };

    struct BucketCache {
        Bitboard Boards[2];
        Accumulator accumulator;
    };
}



#endif // !BITBOARD_H