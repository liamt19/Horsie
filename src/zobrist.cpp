
#include "zobrist.h"

#include "bitboard.h"
#include "types.h"

#define LIZARD_HASHES 1

#if !defined(LIZARD_HASHES)
#include <random>
#endif

namespace Zobrist {

    u64 ColorPieceSquareHashes[2][6][64];
    u64 CastlingRightsHashes[4];
    u64 EnPassantFileHashes[8];
    u64 BlackHash;

    void init() {
#if defined(LIZARD_HASHES)
        for (i32 i = 0; i < 2; i++)
            for (i32 j = 0; j < 6; j++)
                for (i32 k = 0; k < 64; k++)
                    ColorPieceSquareHashes[i][j][k] = LizardPSQT[i][j][k];

        for (i32 i = 0; i < 4; i++)
            CastlingRightsHashes[i] = LizardCR[i];

        for (i32 i = 0; i < 8; i++)
            EnPassantFileHashes[i] = LizardEP[i];

        BlackHash = LizardBH;
#else
        std::mt19937_64 rng(DefaultSeed);
        std::uniform_int_distribution<u64> dis;

        for (i32 i = 0; i < 2; i++)
            for (i32 j = 0; j < 6; j++)
                for (i32 k = 0; k < 64; k++)
                    ColorPieceSquareHashes[i][j][k] = dis(rng);

        for (i32 i = 0; i < 4; i++)
            CastlingRightsHashes[i] = dis(rng);

        for (i32 i = 0; i < 8; i++)
            EnPassantFileHashes[i] = dis(rng);

        BlackHash = dis(rng);
#endif
    }

    void Castle(u64& hash, CastlingStatus prev, CastlingStatus toRemove) {
        u64 change = static_cast<u64>(prev & toRemove);
        while (change != 0) {
            hash ^= CastlingRightsHashes[poplsb(change)];
        }
    }

    void Move(u64& hash, i32 from, i32 to, i32 color, i32 pt) {
        hash ^= ColorPieceSquareHashes[color][pt][from] ^ ColorPieceSquareHashes[color][pt][to];
    }

    void ToggleSquare(u64& hash, i32 color, i32 pt, i32 idx) {
        hash ^= ColorPieceSquareHashes[color][pt][idx];
    }

    void EnPassant(u64& hash, i32 file) {
        hash ^= EnPassantFileHashes[file];
    }

    void ChangeToMove(u64& hash) {
        hash ^= BlackHash;
    }

    u64 GetHash(Horsie::Position& position, u64* pawnHash, u64* nonPawnHash) {
        u64 hash = 0;

        Horsie::Bitboard& bb = position.bb;
        u64 white = bb.Colors[Color::WHITE];
        u64 black = bb.Colors[Color::BLACK];

        while (white != 0) {
            i32 idx = poplsb(white);
            i32 pt = bb.GetPieceAtIndex(idx);
            hash ^= ColorPieceSquareHashes[Color::WHITE][pt][idx];

            if (pt == PAWN) {
                *pawnHash ^= ColorPieceSquareHashes[Color::WHITE][pt][idx];
            }
            else {
                *nonPawnHash ^= ColorPieceSquareHashes[Color::WHITE][pt][idx];
            }
        }

        while (black != 0) {
            i32 idx = poplsb(black);
            i32 pt = bb.GetPieceAtIndex(idx);
            hash ^= ColorPieceSquareHashes[Color::BLACK][pt][idx];

            if (pt == PAWN) {
                *pawnHash ^= ColorPieceSquareHashes[Color::BLACK][pt][idx];
            }
            else {
                *nonPawnHash ^= ColorPieceSquareHashes[Color::BLACK][pt][idx];
            }
        }

        if ((position.State->CastleStatus & CastlingStatus::WK) != CastlingStatus::None) {
            hash ^= CastlingRightsHashes[0];
        }
        if ((position.State->CastleStatus & CastlingStatus::WQ) != CastlingStatus::None) {
            hash ^= CastlingRightsHashes[1];
        }
        if ((position.State->CastleStatus & CastlingStatus::BK) != CastlingStatus::None) {
            hash ^= CastlingRightsHashes[2];
        }
        if ((position.State->CastleStatus & CastlingStatus::BQ) != CastlingStatus::None) {
            hash ^= CastlingRightsHashes[3];
        }

        if (position.State->EPSquare != static_cast<i32>(Square::EP_NONE)) {
            hash ^= EnPassantFileHashes[Horsie::GetIndexFile(position.State->EPSquare)];
        }

        if (position.ToMove == Color::BLACK) {
            hash ^= BlackHash;
        }

        return hash;
    }
}
