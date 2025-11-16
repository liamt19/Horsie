#pragma once

#include "bitboard.h"
#include "defs.h"
#include "enums.h"
#include "move.h"
#include "nnue/accumulator.h"
#include "types.h"
#include "util/list.h"

namespace Horsie {

    class Position {
    public:
        Position(const std::string& fen = InitialFEN);
        void LoadFromFEN(const std::string& fen);

        NNUE::AccumulatorStack Accumulators;
        NNUE::BucketCache CachedBuckets;
        StateInfo State;

        Bitboard bb{};
        Color ToMove{};
        i32 FullMoves{};

        std::array<i32, 15> CastlingRookSquares = {};
        std::array<u64, 15> CastlingRookPaths = {};

        bool IsChess960{};

        constexpr bool InCheck()       const { return State.Checkers != 0; }
        constexpr bool InDoubleCheck() const { return MoreThanOne(State.Checkers); }
        constexpr StateInfo& StartingState() { return States[0]; }

        constexpr u64 CheckSquares(i32 pt) const { return State.CheckSquares[pt]; }
        constexpr u64 BlockingPieces(i32 pc) const { return State.BlockingPieces[pc]; }
        constexpr u64 Pinners(i32 pc) const { return State.Pinners[pc]; }
        constexpr i32 KingSquare(i32 pc) const { return State.KingSquares[pc]; }
        constexpr u64 Checkers() const { return State.Checkers; }
        constexpr u64 Hash() const { return State.Hash; }
        constexpr u64 PawnHash() const { return State.PawnHash; }
        constexpr u64 NonPawnHash(i32 pc) const { return State.NonPawnHash[pc]; }
        constexpr i32 HalfmoveClock() const { return State.HalfmoveClock; }
        constexpr i32 EPSquare() const { return State.EPSquare; }
        constexpr i32 CapturedPiece() const { return State.CapturedPiece; }
        constexpr auto CastleStatus() const { return State.CastleStatus; }

        constexpr auto CurrAccumulator() { return Accumulators.Head(); }
        constexpr auto NextAccumulator() { return Accumulators.Next(); }

        constexpr bool GivesCheck(i32 pt, i32 sq) const { return (CheckSquares(pt) & SquareBB(sq)); }

        constexpr bool CanCastle(u64 boardOcc, u64 ourOcc, CastlingStatus cr) const {
            return HasCastlingRight(cr) && !CastlingImpeded(boardOcc, cr) && HasCastlingRook(ourOcc, cr);
        }

        constexpr i32 CastlingRookSquare(CastlingStatus cr) const { return CastlingRookSquares[static_cast<i32>(cr)]; }
        constexpr u64 CastlingRookPath(CastlingStatus cr) const { return CastlingRookPaths[static_cast<i32>(cr)]; }

        constexpr bool HasCastlingRight(CastlingStatus cr) const { return ((State.CastleStatus & cr) != CastlingStatus::None); }
        constexpr bool CastlingImpeded(u64 boardOcc, CastlingStatus cr) const { return (boardOcc & CastlingRookPath(cr)); }
        constexpr bool HasCastlingRook(u64 ourOcc, CastlingStatus cr) const { return (bb.Pieces[ROOK] & SquareBB(CastlingRookSquare(cr)) & ourOcc); }
        constexpr bool HasNonPawnMaterial(i32 pc) const { return (((bb.Occupancy ^ bb.Pieces[PAWN] ^ bb.Pieces[KING]) & bb.Colors[pc])); }

        constexpr bool IsCapture(Move m) const { return bb.GetPieceAtIndex(m.To()) != Piece::NONE && !m.IsCastle(); }
        constexpr bool IsNoisy(Move m) const { return IsCapture(m) || m.IsEnPassant(); }

        void RemoveCastling(CastlingStatus cr);
        void UpdateHash(i32 pc, i32 pt, i32 sq);
        constexpr CastlingStatus GetCastlingForRook(i32 sq) const;
        Move TryFindMove(const std::string& moveStr, bool& found) const;

        template<bool UpdateNN>
        void MakeMove(Move move);
        void MakeMove(Move move);

        template<bool UpdateNN>
        void UnmakeMove(Move move);
        void UnmakeMove(Move move);

        void MakeNullMove();
        void UnmakeNullMove();

        void DoCastling(i32 ourColor, i32 from, i32 to, bool undo);

        void SetState();
        void SetCheckInfo();
        void CalculateThreats();
        void SetCastlingStatus(i32 c, i32 rfrom);

        u64 HashAfter(Move move) const;

        bool IsPseudoLegal(Move move) const;
        bool IsLegal(Move move) const;
        bool IsLegal(Move move, i32 ourKing, i32 theirKing, u64 pinnedPieces) const;

        bool IsDraw(i16 ply = 0) const;
        bool HasLegalMoves() const;

        i32 MaterialCount() const;

        u64 Perft(i32 depth);
        u64 SplitPerft(i32 depth);

        void SetupForDFRC(i32 wIdx, i32 bIdx);
        std::string GetFEN() const;
        bool SEE_GE(Move m, i32 threshold = 1) const;
        bool HasCycle(i32 ply) const;

        template<i32 pt>
        constexpr u64 ThreatsBy() const {
            switch (pt) {
            case PAWN:   return Threats.PawnThreats;
            case HORSIE: return Threats.HorsieThreats;
            case BISHOP: return Threats.BishopThreats;
            case ROOK:   return Threats.RookThreats;
            case QUEEN:  return Threats.QueenThreats;
            case KING:   return Threats.KingThreats;
            default: return 0;
            };
        }

        constexpr u64 GetAllThreats() const {
            return (Threats.PawnThreats | Threats.HorsieThreats | Threats.BishopThreats | Threats.RookThreats | Threats.QueenThreats | Threats.KingThreats);
        }

    private:
        Util::List<StateInfo, 2048> States{};
        Util::List<Threats, 2048> ThreatsHistory{};
        std::vector<u64> Hashes{};
        Threats Threats;

    };

    std::ostream& operator<<(std::ostream& os, const Position& pos);

}
