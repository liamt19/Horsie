#pragma once

#ifndef ZOBRIST_H
#define ZOBRIST_H

#include "position.h"
#include "types.h"
#include "bitboard.h"
#include <random>


namespace Zobrist {

    //constexpr CastlingStatus operator&(CastlingStatus l, CastlingStatus r) { return l & r; }

    extern ulong ColorPieceSquareHashes[COLOR_NB][PIECE_NB][SQUARE_NB];
    extern ulong CastlingRightsHashes[4];
    extern ulong EnPassantFileHashes[FILE_NB];
    extern ulong BlackHash;
    const int DefaultSeed = 0xBEEF;

    void init();
    void Castle(ulong& hash, CastlingStatus prev, CastlingStatus toRemove);
    void Move(ulong& hash, int from, int to, int color, int pt);
    void ToggleSquare(ulong& hash, int color, int pt, int idx);
    void EnPassant(ulong& hash, int file);
    void ChangeToMove(ulong& hash);
    ulong GetHash(Horsie::Position& position);
}

#endif
