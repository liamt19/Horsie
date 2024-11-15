#pragma once

#ifndef POSITION_H
#define POSITION_H

#include "types.h"
#include "move.h"
#include "nnue/accumulator.h"
#include "bitboard.h"
#include "search.h"
#include "zobrist.h"

#include "nnue/nn.h"
#include "nnue/arch.h"

constexpr int StateStackSize = 1024;

namespace Horsie {

    class Position
    {
    public:
        Position(const std::string& fen = InitialFEN);
        ~Position();
        void LoadFromFEN(const std::string& fen);

        Bitboard bb;
        Color ToMove;
        int FullMoves;
        int GamePly;


        StateInfo* State;
        std::array<BucketCache, InputBuckets * 2> CachedBuckets;

        bool UpdateNN;

        int CastlingRookSquares[(int) CastlingStatus::All];
        ulong CastlingRookPaths[(int) CastlingStatus::All];

        bool IsChess960;
        int dbg_ThisPositionNumber;

        constexpr bool InCheck()       const { return popcount(State->Checkers) == 1; }
        constexpr bool InDoubleCheck() const { return popcount(State->Checkers) == 2; }
        constexpr bool Checked()       const { return popcount(State->Checkers) != 0; }
        constexpr StateInfo* StartingState() const { return _SentinelStart; }
        constexpr StateInfo* PreviousState() const { return State - 1; }
        constexpr StateInfo* NextState() const { return State + 1; }

        constexpr ulong Hash() const { return State->Hash; }
        constexpr ulong PawnHash() const { return State->PawnHash; }
        constexpr ulong NonPawnHash(int pc) const { return State->NonPawnHash[pc]; }
        
        constexpr bool CanCastle(ulong boardOcc, ulong ourOcc, CastlingStatus cr) const {
            return HasCastlingRight(cr) && !CastlingImpeded(boardOcc, cr) && HasCastlingRook(ourOcc, cr);
        }

        constexpr bool HasCastlingRight(CastlingStatus cr) const { return ((State->CastleStatus & cr) != CastlingStatus::None); }
        constexpr bool CastlingImpeded(ulong boardOcc, CastlingStatus cr) const { return (boardOcc & CastlingRookPaths[(int)cr]); }
        constexpr bool HasCastlingRook(ulong ourOcc, CastlingStatus cr) const { return (bb.Pieces[ROOK] & SquareBB(CastlingRookSquares[(int)cr]) & ourOcc); }
        constexpr bool HasNonPawnMaterial(int pc) const { return (((bb.Occupancy ^ bb.Pieces[PAWN] ^ bb.Pieces[KING]) & bb.Colors[pc])); }
        constexpr bool IsCapture(Move m) const { return ((bb.GetPieceAtIndex(m.To()) != Piece::NONE && !m.IsCastle()) || m.IsEnPassant()); }

        void RemoveCastling(CastlingStatus cr) const;
        void UpdateHash(int pc, int pt, int sq) const;
        constexpr CastlingStatus GetCastlingForRook(int sq) const;
        Move TryFindMove(const std::string& moveStr, bool& found) const;

        template<bool UpdateNN>
        void MakeMove(Move move);
        void MakeMove(Move move);

        void UnmakeMove(Move move);
        void MakeNullMove();
        void UnmakeNullMove();

        void DoCastling(int ourColor, int from, int to, bool undo);

        void SetState();
        void SetCheckInfo();
        void SetCastlingStatus(int c, int rfrom);

        ulong HashAfter(Move m) const;

        bool IsPseudoLegal(Move move) const;
        bool IsLegal(Move move) const;
        bool IsLegal(Move move, int ourKing, int theirKing, ulong pinnedPieces) const;

        bool IsDraw() const;
        bool IsInsufficientMaterial() const;
        bool IsThreefoldRepetition() const;
        bool IsFiftyMoveDraw() const;

        ulong Perft(int depth);
        ulong DebugPerft(int depth);
        ulong SplitPerft(int depth);

        std::string GetFEN() const;
        bool SEE_GE(Move m, int threshold = 1) const;
        bool HasCycle(int ply) const;


        inline int PawnCorrectionIndex(int pc) const {
            return (pc * 16384) + (int)(PawnHash() % 16384);
        }

        inline int NonPawnCorrectionIndex(int pc, int side) const {
            return (pc * 16384) + (int)(NonPawnHash(side) % 16384);
        }

    private:
        
        Accumulator* _accumulatorBlock;
        StateInfo* _stateBlock;

        StateInfo* _SentinelStart;
        StateInfo* _SentinelEnd;
    };

    std::ostream& operator<<(std::ostream& os, const Position& pos);

}


#endif