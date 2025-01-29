
#include "bitboard.h"

#include "defs.h"
#include "enums.h"
#include "precomputed.h"
#include "types.h"

#include <cassert>
#include <cstring>

namespace Horsie {

    Bitboard::Bitboard() {
        Reset();
    }

    void Bitboard::CopyTo(Bitboard& other) const {
        std::memcpy(other.Pieces, Pieces, sizeof(u64) * PIECE_NB);
        std::memcpy(other.Colors, Colors, sizeof(u64) * COLOR_NB);
    }

    void Bitboard::Reset() {
        for (i32 i = 0; i < 6; i++)
            Pieces[i] = 0;

        for (i32 i = 0; i < 2; i++)
            Colors[i] = 0;

        for (i32 i = 0; i < 64; i++)
            PieceTypes[i] = Piece::NONE;

        Occupancy = 0;
    }

    void Bitboard::AddPiece(i32 idx, i32 pc, i32 pt) {
        PieceTypes[idx] = pt;

        assert((Colors[pc] & SquareBB(idx)) == 0);
        assert((Pieces[pt] & SquareBB(idx)) == 0);

        Colors[pc] ^= SquareBB(idx);
        Pieces[pt] ^= SquareBB(idx);

        Occupancy |= SquareBB(idx);
    }

    void Bitboard::RemovePiece(i32 idx, i32 pc, i32 pt) {
        PieceTypes[idx] = Piece::NONE;

        assert((Colors[pc] & SquareBB(idx)) != 0);
        assert((Pieces[pt] & SquareBB(idx)) != 0);

        Colors[pc] ^= SquareBB(idx);
        Pieces[pt] ^= SquareBB(idx);

        Occupancy ^= SquareBB(idx);
    }

    void Bitboard::MoveSimple(i32 from, i32 to, i32 pieceColor, i32 pieceType) {
        RemovePiece(from, pieceColor, pieceType);
        AddPiece(to, pieceColor, pieceType);
    }

    i32 Bitboard::KingIndex(i32 pc) const {
        assert(lsb(KingMask(pc)) != SQUARE_NB);
        return lsb(KingMask(pc));
    }

    u64 Bitboard::BlockingPieces(i32 pc, u64* pinners) const {
        u64 blockers = 0UL;
        *pinners = 0;

        u64 temp;
        const u64 us = Colors[pc];
        const u64 them = Colors[Not(pc)];

        const i32 ourKing = KingIndex(pc);

        //  Candidates are their pieces that are on the same rank/file/diagonal as our king.
        u64 candidates = ((RookRays[ourKing] & (Pieces[Piece::QUEEN] | Pieces[Piece::ROOK]))
                          | (BishopRays[ourKing] & (Pieces[Piece::QUEEN] | Pieces[Piece::BISHOP]))) & them;

        const u64 occ = us | them;

        while (candidates != 0) {
            i32 idx = poplsb(candidates);

            temp = BetweenBB[ourKing][idx] & occ;

            if (temp != 0 && !MoreThanOne(temp)) {
                //  If there is one and only one piece between the candidate and our king, that piece is a blocker
                blockers |= temp;

                if ((temp & us) != 0) {
                    //  If the blocker is ours, then the candidate on the square "idx" is a pinner
                    *pinners |= SquareBB(idx);
                }
            }
        }

        return blockers;
    }

    u64 Bitboard::AttackersTo(i32 idx, u64 occupied) const {
        return (attacks_bb<BISHOP>(idx, occupied) & (Pieces[BISHOP] | Pieces[QUEEN]))
            | (attacks_bb<ROOK>(idx, occupied) & (Pieces[ROOK] | Pieces[QUEEN]))
            | (PseudoAttacks[HORSIE][idx] & Pieces[HORSIE])
            | (PawnAttackMasks[WHITE][idx] & Colors[BLACK] & Pieces[PAWN])
            | (PawnAttackMasks[BLACK][idx] & Colors[WHITE] & Pieces[PAWN]);
    }

    u64 Bitboard::AttackMask(i32 idx, i32 pc, i32 pt, u64 occupied) const {
        switch (pt) {
        case PAWN: return PawnAttackMasks[pc][idx];
        case HORSIE: return PseudoAttacks[HORSIE][idx];
        case BISHOP: return attacks_bb<BISHOP>(idx, occupied);
        case ROOK: return attacks_bb<ROOK>(idx, occupied);
        case QUEEN: return attacks_bb<QUEEN>(idx, occupied);
        case KING: return PseudoAttacks[KING][idx];
        default:
            return 0;
        };
    }
}
