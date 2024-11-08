
#include "bitboard.h"
#include "types.h"
#include "precomputed.h"

#define KnightMasks horsie_masks
#define GetBishopMoves get_bishop_moves
#define GetRookMoves get_rook_moves

namespace Horsie {

    Bitboard::Bitboard() {
        Reset();
    }

    void Bitboard::CopyTo(Bitboard& other) const {
        CopyBlock(other.Pieces, Pieces, sizeof(ulong) * PIECE_NB);
        CopyBlock(other.Colors, Colors, sizeof(ulong) * COLOR_NB);
    }

    void Bitboard::Reset()
    {
        for (int i = 0; i < 6; i++)
            Pieces[i] = 0;

        for (int i = 0; i < 2; i++)
            Colors[i] = 0;

        for (int i = 0; i < 64; i++)
            PieceTypes[i] = Piece::NONE;

        Occupancy = 0;
    }


    void Bitboard::AddPiece(int idx, int pc, int pt)
    {
        PieceTypes[idx] = pt;

        assert((Colors[pc] & SquareBB(idx)) == 0);
        assert((Pieces[pt] & SquareBB(idx)) == 0);

        Colors[pc] ^= SquareBB(idx);
        Pieces[pt] ^= SquareBB(idx);

        Occupancy |= SquareBB(idx);
    }

    void Bitboard::RemovePiece(int idx, int pc, int pt)
    {
        PieceTypes[idx] = Piece::NONE;

        assert((Colors[pc] & SquareBB(idx)) != 0);
        assert((Pieces[pt] & SquareBB(idx)) != 0);

        Colors[pc] ^= SquareBB(idx);
        Pieces[pt] ^= SquareBB(idx);

        Occupancy ^= SquareBB(idx);
    }

    void Bitboard::MoveSimple(int from, int to, int pieceColor, int pieceType)
    {
        RemovePiece(from, pieceColor, pieceType);
        AddPiece(to, pieceColor, pieceType);
    }



    int Bitboard::KingIndex(int pc) const
    {
        assert((int) lsb(KingMask(pc)) != SQUARE_NB);

        return (int) lsb(KingMask(pc));
    }

    int Bitboard::MaterialCount(int pc, bool excludePawns) const
    {
        int mat = 0;
        ulong temp = Colors[pc];
        while (temp != 0)
        {
            int idx = poplsb(temp);

            int pt = GetPieceAtIndex((int)idx);
            if (!(excludePawns && pt == Piece::PAWN))
            {
                mat += GetPieceValue(Piece(pt));
            }
        }

        return mat;
    }

    ulong Bitboard::BlockingPieces(int pc, ulong* pinners, ulong* xrayers) const
    {
        ulong blockers = 0UL;
        *pinners = 0;
        *xrayers = 0;

        ulong temp;
        ulong us = Colors[pc];
        ulong them = Colors[Not(pc)];

        int ourKing = KingIndex(pc);

        //  Candidates are their pieces that are on the same rank/file/diagonal as our king.
        ulong candidates = ((RookRays[ourKing] & (Pieces[Piece::QUEEN] | Pieces[Piece::ROOK]))
            | (BishopRays[ourKing] & (Pieces[Piece::QUEEN] | Pieces[Piece::BISHOP]))) & them;

        ulong occ = us | them;

        while (candidates != 0)
        {
            int idx = poplsb(candidates);

            temp = BetweenBB[ourKing][(int)idx] & occ;

            if (temp != 0 && !MoreThanOne(temp))
            {
                //  If there is one and only one piece between the candidate and our king, that piece is a blocker
                blockers |= temp;

                if ((temp & us) != 0)
                {
                    //  If the blocker is ours, then the candidate on the square "idx" is a pinner
                    *pinners |= SquareBB(idx);
                }
                else
                {
                    //  If the blocker isn't ours, then it will cause a discovered check if it moves
                    *xrayers |= SquareBB(idx);
                }
            }
        }

        return blockers;
    }

    ulong Bitboard::AttackersTo(int idx, ulong occupied) const
    {
        return (attacks_bb<BISHOP>(idx, occupied) & (Pieces[BISHOP] | Pieces[QUEEN]))
            | (attacks_bb<ROOK>(idx, occupied) & (Pieces[ROOK] | Pieces[QUEEN]))
            | (PseudoAttacks[HORSIE][(int)idx] & Pieces[HORSIE])
            | (PawnAttackMasks[WHITE][(int)idx] & Colors[BLACK] & Pieces[PAWN])
            | (PawnAttackMasks[BLACK][(int)idx] & Colors[WHITE] & Pieces[PAWN]);
    }

    ulong Bitboard::AttackMask(int idx, int pc, int pt, ulong occupied) const
    {
        switch (pt)
        {
        case Pawn: return PawnAttackMasks[pc][(int)idx];
        case Knight: return PseudoAttacks[HORSIE][(int)idx];
        case Bishop: return attacks_bb<BISHOP>(idx, occupied);
        case Rook: return attacks_bb<ROOK>(idx, occupied);
        case Queen: return attacks_bb<QUEEN>(idx, occupied);
        case King: return PseudoAttacks[KING][(int)idx];
        default:
            return 0;
        };
    }

}
