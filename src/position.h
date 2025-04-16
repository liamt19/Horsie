#pragma once

#include "bitboard.h"
#include "defs.h"
#include "enums.h"
#include "eval/material.h"
#include "move.h"
#include "types.h"

namespace Horsie {

    class Position {
    public:
        Position(const std::string& fen = InitialFEN);
        ~Position();
        void LoadFromFEN(const std::string& fen);

        StateInfo* State;

        Bitboard bb;
        Color ToMove;
        i32 FullMoves;
        i32 GamePly;

        Eval::MaterialScore MatScore{};

        i32 CastlingRookSquares[static_cast<i32>(CastlingStatus::All)];
        u64 CastlingRookPaths[static_cast<i32>(CastlingStatus::All)];

        bool UpdateNN;
        bool IsChess960;

        constexpr bool InCheck()       const { return State->Checkers != 0; }
        constexpr bool InDoubleCheck() const { return MoreThanOne(State->Checkers); }
        constexpr StateInfo* StartingState() const { return _SentinelStart; }
        constexpr StateInfo* PreviousState() const { return State - 1; }
        constexpr StateInfo* NextState() const { return State + 1; }

        constexpr u64 CheckSquares(i32 pt) const { return State->CheckSquares[pt]; }
        constexpr u64 BlockingPieces(i32 pc) const { return State->BlockingPieces[pc]; }
        constexpr u64 Pinners(i32 pc) const { return State->Pinners[pc]; }
        constexpr i32 KingSquare(i32 pc) const { return State->KingSquares[pc]; }
        constexpr u64 Checkers() const { return State->Checkers; }
        constexpr u64 Hash() const { return State->Hash; }
        constexpr u64 PawnHash() const { return State->PawnHash; }
        constexpr u64 NonPawnHash(i32 pc) const { return State->NonPawnHash[pc]; }
        constexpr i32 HalfmoveClock() const { return State->HalfmoveClock; }
        constexpr i32 EPSquare() const { return State->EPSquare; }
        constexpr i32 CapturedPiece() const { return State->CapturedPiece; }
        constexpr auto CastleStatus() const { return State->CastleStatus; }

        constexpr bool GivesCheck(i32 pt, i32 sq) const { return (CheckSquares(pt) & SquareBB(sq)); }

        constexpr bool CanCastle(u64 boardOcc, u64 ourOcc, CastlingStatus cr) const {
            return HasCastlingRight(cr) && !CastlingImpeded(boardOcc, cr) && HasCastlingRook(ourOcc, cr);
        }

        constexpr i32 CastlingRookSquare(CastlingStatus cr) const { return CastlingRookSquares[static_cast<i32>(cr)]; }
        constexpr u64 CastlingRookPath(CastlingStatus cr) const { return CastlingRookPaths[static_cast<i32>(cr)]; }

        constexpr bool HasCastlingRight(CastlingStatus cr) const { return ((State->CastleStatus & cr) != CastlingStatus::None); }
        constexpr bool CastlingImpeded(u64 boardOcc, CastlingStatus cr) const { return (boardOcc & CastlingRookPath(cr)); }
        constexpr bool HasCastlingRook(u64 ourOcc, CastlingStatus cr) const { return (bb.Pieces[ROOK] & SquareBB(CastlingRookSquare(cr)) & ourOcc); }
        constexpr bool HasNonPawnMaterial(i32 pc) const { return (((bb.Occupancy ^ bb.Pieces[PAWN] ^ bb.Pieces[KING]) & bb.Colors[pc])); }

        constexpr bool IsCapture(Move m) const { return bb.GetPieceAtIndex(m.To()) != Piece::NONE && !m.IsCastle(); }
        constexpr bool IsNoisy(Move m) const { return IsCapture(m) || m.IsEnPassant(); }

        void RemoveCastling(CastlingStatus cr) const;
        void UpdateHash(i32 pc, i32 pt, i32 sq) const;
        constexpr CastlingStatus GetCastlingForRook(i32 sq) const;
        Move TryFindMove(const std::string& moveStr, bool& found) const;

        void MakeMove(Move move);
        void UnmakeMove(Move move);
        void MakeNullMove();
        void UnmakeNullMove();

        void AddPiece(i32 idx, i32 pc, i32 pt);
        void RemovePiece(i32 idx, i32 pc, i32 pt);
        void MoveSimple(i32 from, i32 to, i32 pc, i32 pt);

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
        bool HasLegalMoves() const;

        i32 MaterialCount() const;

        u64 Perft(i32 depth);
        u64 SplitPerft(i32 depth);

        void SetupForDFRC(i32 wIdx, i32 bIdx);
        std::string GetFEN() const;
        bool SEE_GE(Move m, i32 threshold = 1) const;
        bool HasCycle(i32 ply) const;

        template<i32 pt>
        constexpr u64 ThreatsBy(i32 pc) const {
            return bb.ThreatsBy<pt>(pc);
        }

        inline i32 PawnCorrectionIndex(i32 pc) const {
            return (pc * 16384) + static_cast<i32>(PawnHash() % 16384);
        }

        inline i32 NonPawnCorrectionIndex(i32 pc, i32 side) const {
            return (pc * 16384) + static_cast<i32>(NonPawnHash(side) % 16384);
        }

        inline auto GetMaterialScore() const {
            const auto material = MatScore.get();
            return ToMove == BLACK ? -material : material;
        }

    private:
        static constexpr i32 StateStackSize = 1024; 

        StateInfo* _stateBlock;

        StateInfo* _SentinelStart;
        StateInfo* _SentinelEnd;
    };

    std::ostream& operator<<(std::ostream& os, const Position& pos);

}
