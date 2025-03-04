#pragma once

#include "bitboard.h"
#include "defs.h"
#include "enums.h"
#include "move.h"
#include "nnue/accumulator.h"
#include "nnue/arch.h"
#include "types.h"

constexpr i32 StateStackSize = 1024;

namespace Horsie {

    class Position {
    public:
        Position(const std::string& fen = InitialFEN);
        ~Position();

        Bitboard bb;
        Color ToMove;
        i32 FullMoves;
        i32 GamePly;

        StateInfo* State;
        std::array<BucketCache, INPUT_BUCKETS * 2> CachedBuckets;

        bool IsChess960;

        constexpr StateInfo* StartingState() const { return _stateBlock; }
        constexpr StateInfo* PreviousState() const { return State - 1; }
        constexpr StateInfo* NextState() const { return State + 1; }

        constexpr bool InCheck()       const { return popcount(State->Checkers) == 1; }
        constexpr bool InDoubleCheck() const { return popcount(State->Checkers) == 2; }
        constexpr bool Checked()       const { return popcount(State->Checkers) != 0; }

        constexpr u64 CheckSquares(i32 pt) const { return State->CheckSquares[pt]; }
        constexpr u64 BlockingPieces(i32 pc) const { return State->BlockingPieces[pc]; }
        constexpr i32 KingSquare(i32 pc) const { return State->KingSquares[pc]; }
        constexpr u64 Checkers() const { return State->Checkers; }
        constexpr u64 Hash() const { return State->Hash; }
        constexpr u64 PawnHash() const { return State->PawnHash; }
        constexpr u64 NonPawnHash(i32 pc) const { return State->NonPawnHash[pc]; }
        constexpr i32 HalfmoveClock() const { return State->HalfmoveClock; }
        constexpr i32 EPSquare() const { return State->EPSquare; }
        constexpr i32 CapturedPiece() const { return State->CapturedPiece; }

        constexpr i32 CastlingSquare(CastlingStatus cr) const { return CastlingRookSquares[static_cast<i32>(cr)]; }

        constexpr bool HasCastlingRight(CastlingStatus cr) const { return ((State->CastleStatus & cr) != CastlingStatus::None); }
        constexpr bool CastlingImpeded(u64 boardOcc, CastlingStatus cr) const { return (boardOcc & CastlingRookPaths[static_cast<i32>(cr)]); }
        constexpr bool HasCastlingRook(u64 ourOcc, CastlingStatus cr) const { return (bb.Pieces[ROOK] & SquareBB(CastlingSquare(cr)) & ourOcc); }
        constexpr bool HasNonPawnMaterial(i32 pc) const { return (((bb.Occupancy ^ bb.Pieces[PAWN] ^ bb.Pieces[KING]) & bb.Colors[pc])); }

        constexpr bool CanCastle(u64 boardOcc, u64 ourOcc, CastlingStatus cr) const {
            return HasCastlingRight(cr) && !CastlingImpeded(boardOcc, cr) && HasCastlingRook(ourOcc, cr);
        }

        constexpr bool IsCapture(Move m) const { return bb.GetPieceAtIndex(m.To()) != Piece::NONE && !m.IsCastle(); }
        constexpr bool IsNoisy(Move m) const { return IsCapture(m) || m.IsEnPassant(); }

        void RemoveCastling(CastlingStatus cr) const;
        void UpdateHash(i32 pc, i32 pt, i32 sq) const;
        constexpr CastlingStatus GetCastlingForRook(i32 sq) const;
        Move TryFindMove(const std::string& moveStr, bool& found) const;

        template<bool UpdateNN>
        void MakeMove(Move move);
        void MakeMove(Move move);

        void UnmakeMove(Move move);

        void MakeNullMove();
        void UnmakeNullMove();

        void DoCastling(i32 ourColor, i32 from, i32 to, bool undo);

        void SetState();
        void SetCheckInfo();
        void SetCastlingStatus(i32 c, i32 rfrom);

        u64 HashAfter(Move move) const;

        bool IsPseudoLegal(Move move) const;
        bool IsLegal(Move move) const;
        bool IsLegal(Move move, i32 ourKing, i32 theirKing, u64 pinnedPieces) const;

        bool IsDraw() const;
        bool IsInsufficientMaterial() const;
        bool IsThreefoldRepetition() const;
        bool IsFiftyMoveDraw() const;

        u64 Perft(i32 depth);
        u64 SplitPerft(i32 depth);

        void LoadFromFEN(const std::string& fen);
        std::string GetFEN() const;

        bool HasCycle(i32 ply) const;
        void SetupForDFRC(i32 wIdx, i32 bIdx);
        bool HasLegalMoves() const;
        bool SEE_GE(Move m, i32 threshold = 1) const;

        inline i32 PawnCorrectionIndex(i32 pc) const { return (pc * 16384) + static_cast<i32>(PawnHash() % 16384); }
        inline i32 NonPawnCorrectionIndex(i32 pc, i32 side) const { return (pc * 16384) + static_cast<i32>(NonPawnHash(side) % 16384); }
        
        i32 CastlingRookSquares[static_cast<i32>(CastlingStatus::All)];
        u64 CastlingRookPaths[static_cast<i32>(CastlingStatus::All)];

    private:

        Accumulator* _accumulatorBlock;
        StateInfo* _stateBlock;


    };

    std::ostream& operator<<(std::ostream& os, const Position& pos);

}
