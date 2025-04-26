
#include "zobrist.h"

#include "bitboard.h"
#include "types.h"

#define LIZARD_HASHES 1

#if !defined(LIZARD_HASHES)
#include <random>
#endif

namespace Horsie::Zobrist {

    u64 PSQHashes[2][6][64];
    u64 CRHashes[4];
    u64 EPHashes[8];
    u64 BlackHash;

    void Init() {
#if defined(LIZARD_HASHES)
        for (i32 i = 0; i < 2; i++)
            for (i32 j = 0; j < 6; j++)
                for (i32 k = 0; k < 64; k++)
                    PSQHashes[i][j][k] = LizardPSQT[i][j][k];

        for (i32 i = 0; i < 4; i++)
            CRHashes[i] = LizardCR[i];

        for (i32 i = 0; i < 8; i++)
            EPHashes[i] = LizardEP[i];

        BlackHash = LizardBH;
#else
        std::mt19937_64 rng(DefaultSeed);
        std::uniform_int_distribution<u64> dis;

        for (i32 i = 0; i < 2; i++)
            for (i32 j = 0; j < 6; j++)
                for (i32 k = 0; k < 64; k++)
                    PSQHashes[i][j][k] = dis(rng);

        for (i32 i = 0; i < 4; i++)
            CRHashes[i] = dis(rng);

        for (i32 i = 0; i < 8; i++)
            EPHashes[i] = dis(rng);

        BlackHash = dis(rng);
#endif
    }

    void Castle(u64& hash, CastlingStatus prev, CastlingStatus toRemove) {
        u64 change = static_cast<u64>(prev & toRemove);
        while (change != 0) {
            hash ^= CRHashes[poplsb(change)];
        }
    }

    void Move(u64& hash, i32 from, i32 to, i32 color, i32 pt) {
        hash ^= PSQHashes[color][pt][from] ^ PSQHashes[color][pt][to];
    }

    void ToggleSquare(u64& hash, i32 color, i32 pt, i32 idx) {
        hash ^= PSQHashes[color][pt][idx];
    }

    void EnPassant(u64& hash, i32 file) {
        hash ^= EPHashes[file];
    }

    void ChangeToMove(u64& hash) {
        hash ^= BlackHash;
    }

    void SetHashes(Horsie::Position& pos) {
        const auto state = pos.State;
        const auto& bb = pos.bb;

        u64 hash = 0, pawnHash = 0, nonPawnHash = 0;

        for (i32 pc = 0; pc < 2; pc++) {
            u64 pieces = bb.Colors[pc];

            while (pieces != 0) {
                i32 idx = poplsb(pieces);
                i32 pt = bb.GetPieceAtIndex(idx);
                hash ^= PSQHashes[pc][pt][idx];

                if (pt == PAWN) {
                    pawnHash ^= PSQHashes[pc][pt][idx];
                }
                else {
                    nonPawnHash ^= PSQHashes[pc][pt][idx];
                }
            }
        }
        
        if (pos.HasCastlingRight(CastlingStatus::WK))
            hash ^= CRHashes[0];

        if (pos.HasCastlingRight(CastlingStatus::WQ))
            hash ^= CRHashes[1];

        if (pos.HasCastlingRight(CastlingStatus::BK))
            hash ^= CRHashes[2];

        if (pos.HasCastlingRight(CastlingStatus::BQ))
            hash ^= CRHashes[3];

        if (pos.EPSquare() != EP_NONE) {
            hash ^= EPHashes[GetIndexFile(pos.EPSquare())];
        }

        if (pos.ToMove == Color::BLACK) {
            hash ^= BlackHash;
        }

        state->Hash = hash;
        state->PawnHash = pawnHash;
        state->NonPawnHash[WHITE] = state->NonPawnHash[BLACK] = nonPawnHash;
    }
}
