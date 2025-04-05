
#include "zobrist.h"

#include "bitboard.h"
#include "types.h"

#define LIZARD_HASHES 1

#if !defined(LIZARD_HASHES)
#include <random>
#endif

using namespace Horsie;

namespace Zobrist {

    u64 PSQHashes[2][6][64];
    u64 CastlingHashes[4];
    u64 EPHashes[8];
    u64 ColorHash;

    void Init() {
#if defined(LIZARD_HASHES)
        for (i32 i = 0; i < 2; i++)
            for (i32 j = 0; j < 6; j++)
                for (i32 k = 0; k < 64; k++)
                    PSQHashes[i][j][k] = LizardPSQT[i][j][k];

        for (i32 i = 0; i < 4; i++)
            CastlingHashes[i] = LizardCR[i];

        for (i32 i = 0; i < 8; i++)
            EPHashes[i] = LizardEP[i];

        ColorHash = LizardBH;
#else
        std::mt19937_64 rng(DefaultSeed);
        std::uniform_int_distribution<u64> dis;

        for (i32 i = 0; i < 2; i++)
            for (i32 j = 0; j < 6; j++)
                for (i32 k = 0; k < 64; k++)
                    PSQHashes[i][j][k] = dis(rng);

        for (i32 i = 0; i < 4; i++)
            CastlingHashes[i] = dis(rng);

        for (i32 i = 0; i < 8; i++)
            EPHashes[i] = dis(rng);

        ColorHash = dis(rng);
#endif
    }

    void Castle(u64& hash, CastlingStatus prev, CastlingStatus toRemove) {
        u64 change = static_cast<u64>(prev & toRemove);
        while (change != 0) {
            hash ^= CastlingHashes[poplsb(change)];
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
        hash ^= ColorHash;
    }

    u64 GetHash(Position& pos, u64* pawnHash, u64* nonPawnHash) {
        u64 hash = 0;

        Bitboard& bb = pos.bb;

        for (auto pc : {WHITE, BLACK}) {
            auto mask = bb.Colors[pc];
            while (mask != 0) {
                i32 sq = poplsb(mask);
                i32 pt = bb.GetPieceAtIndex(sq);
                hash ^= PSQHashes[pc][pt][sq];

                if (pt == PAWN)
                    *pawnHash ^= PSQHashes[pc][pt][sq];
                else
                    *nonPawnHash ^= PSQHashes[pc][pt][sq];
            }
        }
        
        const auto cr = pos.CastleStatus();
        if ((cr & CastlingStatus::WK) != CastlingStatus::None) hash ^= CastlingHashes[0];
        if ((cr & CastlingStatus::WQ) != CastlingStatus::None) hash ^= CastlingHashes[1];
        if ((cr & CastlingStatus::BK) != CastlingStatus::None) hash ^= CastlingHashes[2];
        if ((cr & CastlingStatus::BQ) != CastlingStatus::None) hash ^= CastlingHashes[3];

        if (pos.EPSquare() != EP_NONE) {
            hash ^= EPHashes[Horsie::GetIndexFile(pos.EPSquare())];
        }

        if (pos.ToMove == Color::BLACK) {
            hash ^= ColorHash;
        }

        return hash;
    }
}
