
#include "zobrist.h"

#define LIZARD_HASHES 1
//#undef LIZARD_HASHES

namespace Zobrist {

    ulong ColorPieceSquareHashes[2][6][64];
    ulong CastlingRightsHashes[4];
    ulong EnPassantFileHashes[8];
    ulong BlackHash;

    void init() {
#if defined(LIZARD_HASHES)
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 6; j++)
                for (int k = 0; k < 64; k++)
                    ColorPieceSquareHashes[i][j][k] = LizardPSQT[i][j][k];

        for (int i = 0; i < 4; i++)
            CastlingRightsHashes[i] = LizardCR[i];

        for (int i = 0; i < 8; i++)
            EnPassantFileHashes[i] = LizardEP[i];

        BlackHash = LizardBH;
#else
        std::mt19937_64 rng(DefaultSeed);
        std::uniform_int_distribution<uint64_t> dis;

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 6; j++)
                for (int k = 0; k < 64; k++)
                    ColorPieceSquareHashes[i][j][k] = dis(rng);

        for (int i = 0; i < 4; i++)
            CastlingRightsHashes[i] = dis(rng);

        for (int i = 0; i < 8; i++)
            EnPassantFileHashes[i] = dis(rng);

        BlackHash = dis(rng);
#endif
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

    ulong GetHash(Horsie::Position& position, ulong* pawnHash, ulong* nonPawnHash) {
        ulong hash = 0;

        Horsie::Bitboard& bb = position.bb;
        ulong white = bb.Colors[Color::WHITE];
        ulong black = bb.Colors[Color::BLACK];

        while (white != 0)
        {
            int idx = (int)poplsb(white);
            int pt = bb.GetPieceAtIndex(idx);
            hash ^= ColorPieceSquareHashes[Color::WHITE][pt][idx];

            if (pt == PAWN) {
                *pawnHash ^= ColorPieceSquareHashes[Color::WHITE][pt][idx];
            }
            else {
                *nonPawnHash ^= ColorPieceSquareHashes[Color::WHITE][pt][idx];
            }
        }

        while (black != 0)
        {
            int idx = (int)poplsb(black);
            int pt = bb.GetPieceAtIndex(idx);
            hash ^= ColorPieceSquareHashes[Color::BLACK][pt][idx];

            if (pt == PAWN) {
                *pawnHash ^= ColorPieceSquareHashes[Color::BLACK][pt][idx];
            }
            else {
                *nonPawnHash ^= ColorPieceSquareHashes[Color::BLACK][pt][idx];
            }
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