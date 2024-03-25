#include "zobrist.h"



namespace Zobrist {

    ulong ColorPieceSquareHashes[COLOR_NB][PIECE_NB][SQUARE_NB];
    ulong CastlingRightsHashes[4];
    ulong EnPassantFileHashes[FILE_NB];
    ulong BlackHash;

    void init() {
        std::mt19937_64 rng(DefaultSeed);
        std::uniform_int_distribution<uint64_t> dis;

        BlackHash = dis(rng);

        for (int i = 0; i < COLOR_NB; i++) {
            for (int j = 0; j < PIECE_NB; j++) {
                for (int k = 0; k < SQUARE_NB; k++) {
                    ColorPieceSquareHashes[i][j][k] = dis(rng);
                }
            }
        }

        for (int i = 0; i < 4; i++)
        {
            CastlingRightsHashes[i] = dis(rng);
        }

        for (int i = 0; i < FILE_NB; i++)
        {
            EnPassantFileHashes[i] = dis(rng);
        }


    }

    void Castle(ulong& hash, CastlingStatus prev, CastlingStatus toRemove) {
        ulong change = (ulong)(prev & toRemove);
        while (change != 0)
        {
            hash ^= CastlingRightsHashes[(int)poplsb(change)];
        }
    }


    void Move(ulong& hash, int from, int to, int color, int pt) {
        hash ^= ColorPieceSquareHashes[color][pt][from] ^ ColorPieceSquareHashes[color][pt][to];
    }

    void ToggleSquare(ulong& hash, int color, int pt, int idx) {
        hash ^= ColorPieceSquareHashes[color][pt][idx];
    }

    void EnPassant(ulong& hash, int file) {
        hash ^= EnPassantFileHashes[file];
    }

    void ChangeToMove(ulong& hash) {
        hash ^= BlackHash;
    }

    ulong GetHash(Horsie::Position& position) {
        ulong hash = 0;

        Horsie::Bitboard bb = position.bb;
        ulong white = bb.Colors[Color::WHITE];
        ulong black = bb.Colors[Color::BLACK];

        while (white != 0)
        {
            int idx = (int)poplsb(white);
            hash ^= ColorPieceSquareHashes[Color::WHITE][bb.GetPieceAtIndex(idx)][idx];
        }

        while (black != 0)
        {
            int idx = (int)poplsb(black);
            hash ^= ColorPieceSquareHashes[Color::BLACK][bb.GetPieceAtIndex(idx)][idx];
        }

        if ((position.State->CastleStatus & CastlingStatus::WK) != CastlingStatus::None)
        {
            hash ^= CastlingRightsHashes[0];
        }
        if ((position.State->CastleStatus & CastlingStatus::WQ) != CastlingStatus::None)
        {
            hash ^= CastlingRightsHashes[1];
        }
        if ((position.State->CastleStatus & CastlingStatus::BK) != CastlingStatus::None)
        {
            hash ^= CastlingRightsHashes[2];
        }
        if ((position.State->CastleStatus & CastlingStatus::BQ) != CastlingStatus::None)
        {
            hash ^= CastlingRightsHashes[3];
        }

        if (position.State->EPSquare != (int)Square::EP_NONE)
        {
            hash ^= EnPassantFileHashes[Horsie::GetIndexFile(position.State->EPSquare)];
        }

        if (position.ToMove == Color::BLACK)
        {
            hash ^= BlackHash;
        }

        return hash;
    }

}