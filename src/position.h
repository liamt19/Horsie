#pragma once

#ifndef POSITION_H
#define POSITION_H

#include "types.h"
#include "move.h"
#include "accumulator.h"
#include "bitboard.h"
#include "search_thread.h"

constexpr int StateStackSize = 2048;

namespace Horsie {

	class Position
	{
	public:
		Position(const std::string& fen, SearchThread* owningThread);
		~Position();
		void LoadFromFEN(const std::string& fen);

		Bitboard bb;
		Color ToMove;
		int MaterialCountNonPawn[2];
		int FullMoves;
		int GamePly;
		bool InCheck;
		bool InDoubleCheck;
		int idxChecker;

		SearchThread Owner;
		StateInfo* State;

		bool UpdateNN;

		int CastlingRookSquares[(int) CastlingStatus::All];
		ulong CastlingRookPaths[(int) CastlingStatus::All];

		bool IsChess960;

		constexpr bool Checked() const { return InCheck || InDoubleCheck; }
		constexpr StateInfo* StartingState() const { return _SentinelStart; }
		constexpr StateInfo* PreviousState() const { return (State == _SentinelStart) ? NULL : _stateBlock - 1; }
		constexpr StateInfo* NextState() const { return (State != _SentinelEnd) ? State + 1 : NULL; }


		void MakeMove(Move move);
		void UnmakeMove(Move move);
		void MakeNullMove();
		void UnmakeNullMove();

		void DoCastling(int ourColor, int from, int to, bool undo);

		void SetState();
		void SetCheckInfo();
		void SetCastlingStatus(int c, int rfrom);

		ulong HashAfter(Move m);

		bool IsPseudoLegal(Move move) const;
		bool IsLegal(Move move) const;
		bool IsLegal(Move move, int ourKing, int theirKing, ulong pinnedPieces) const;

		bool IsDraw() const;
		bool IsInsufficientMaterial() const;
		bool IsThreefoldRepetition() const;
		bool IsFiftyMoveDraw() const;

		ulong Perft(int depth);

		std::string GetFEN() const;
	private:
		
		Accumulator* _accumulatorBlock;
		StateInfo* _stateBlock;

		StateInfo* _SentinelStart;
		StateInfo* _SentinelEnd;
	};

	std::ostream& operator<<(std::ostream& os, const Position& pos);

}


#endif