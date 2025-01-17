#pragma once

#ifndef MOVEPICKER_H
#define MOVEPICKER_H

#include "search.h"
#include "position.h"
#include "nnue/nn.h"
#include "tt.h"
#include "movegen.h"
#include "precomputed.h"
#include "search_options.h"
#include "threadpool.h"


namespace Horsie {

	enum MovePickerStage : i32 {
		NegamaxTT = 0,
		NegamaxKiller,
		NegamaxMakeMoves,
		NegamaxPlayMoves,

		QSearchTT,
		QSearchMakeMoves,
		QSearchPlayMoves,

		End,
	};

	class MovePicker {
	public:
		static auto negamax(Position& pos, SearchStackEntry* ss, HistoryTable& history, Move ttMove) {
			return MovePicker(pos, ss, history, ttMove, MovePickerStage::NegamaxTT);
		}

		static auto qsearch(Position& pos, SearchStackEntry* ss, HistoryTable& history, Move ttMove) {
			return MovePicker(pos, ss, history, ttMove, MovePickerStage::QSearchTT);
		}

		MovePicker(Position& pos, SearchStackEntry* ss, HistoryTable& history, Move ttMove, MovePickerStage stage)
			: pos(pos), ss(ss), history(history), ttMove(ttMove), stage(stage) {
			
			contHist = { (ss - 1)->ContinuationHistory, (ss - 2)->ContinuationHistory,
						 (ss - 4)->ContinuationHistory, (ss - 6)->ContinuationHistory };

			killer = ss->KillerMove;
			if (killer == ttMove) {
				killer = Move::Null();
			}

			listSize = moveIndex = 0;
		}

		Move Next() {

			switch (stage) {

			case NegamaxTT:
			{
				stage++;

				if (ttMove != Move::Null() && pos.IsPseudoLegal(ttMove)) {
					return ttMove;
				}
				else {
					return Next();
				}
			}

			case QSearchTT:
			{
				stage++;

				if (ttMove != Move::Null() 
					&& pos.IsPseudoLegal(ttMove) 
					&& (pos.IsCapture(ttMove) || pos.Checked())) {
					return ttMove;
				}
				else {
					return Next();
				}
			}

			case NegamaxKiller:
			{
				stage++;

				if (killer != Move::Null() && pos.IsPseudoLegal(killer)) {
					return killer;
				}

				[[fallthrough]];
			}

			case NegamaxMakeMoves:
			{
				i32 generated = Generate<PseudoLegal>(pos, list, listSize);
				listSize = generated;
				AssignScores();

				stage++;
				[[fallthrough]];
			}

			case NegamaxPlayMoves:
			{
				if (const auto move = SelectNext([this](Move move) { return move != ttMove && move != killer; })) {
					return move;
				}

				stage = MovePickerStage::End;
				return Move::Null();
			}

			case QSearchMakeMoves:
			{
				i32 generated = GenerateQS(pos, list, listSize);
				listSize = generated;
				AssignQuiescenceScores();

				stage++;
				[[fallthrough]];
			}

			case QSearchPlayMoves:
			{
				if (const auto move = SelectNext([this](Move move) { return move != ttMove; })) {
					return move;
				}

				stage = MovePickerStage::End;
				return Move::Null();
			}

			case End:
			{
				return Move::Null();
			}

			}

			assert(false);
			return Move::Null();
		}

		void StartSkippingQuiets() { skipQuiets = true; }

	private:
		const Position& pos;
		SearchStackEntry* const ss;
		const HistoryTable& history;
		const Move ttMove;
		std::array<PieceToHistory*, 4> contHist;

		Move killer;
		ScoredMove list[MoveListSize] = {};
		i32 listSize;
		i32 moveIndex;

		i32 stage;
		bool skipQuiets;

#if defined(NO)
		const i32 TT_SCORE = -200000000;
		const i32 KILLER_SCORE = -200000000;
#else
		const i32 TT_SCORE = INT32_MAX - 100000;
		const i32 KILLER_SCORE = INT32_MAX - 1000000;
#endif

		void AssignQuiescenceScores() {
			const Bitboard& bb = pos.bb;
			const auto pc = pos.ToMove;

			for (i32 i = moveIndex; i < listSize; i++) {
				Move m = list[i].move;
				const auto [moveFrom, moveTo] = m.Unpack();
				const auto pt = bb.GetPieceAtIndex(moveFrom);

				if (m == ttMove) {
					list[i].Score = TT_SCORE;
				}
				else if (bb.GetPieceAtIndex(moveTo) != Piece::NONE && !m.IsCastle()) {
					const auto capturedPiece = bb.GetPieceAtIndex(moveTo);
					auto& hist = history.CaptureHistory[pc][pt][moveTo][capturedPiece];
					list[i].Score = (MVVMult * GetPieceValue(capturedPiece)) + hist;
				}
				else {
					i32 contIdx = MakePiece(pc, pt);

					list[i].Score  = 2 * history.MainHistory[pc][m.GetMoveMask()];
					list[i].Score += 2 * (*(ss - 1)->ContinuationHistory)[contIdx][moveTo];
					list[i].Score +=	 (*(ss - 2)->ContinuationHistory)[contIdx][moveTo];
					list[i].Score +=	 (*(ss - 4)->ContinuationHistory)[contIdx][moveTo];
					list[i].Score +=	 (*(ss - 6)->ContinuationHistory)[contIdx][moveTo];

					if ((pos.State->CheckSquares[pt] & SquareBB(moveTo)) != 0) {
						list[i].Score += CheckBonus;
					}
				}

				if (pt == HORSIE) {
					list[i].Score += 200;
				}
			}
		}



		void AssignScores() {

			const Bitboard& bb = pos.bb;
			const auto pc = pos.ToMove;

			for (i32 i = moveIndex; i < listSize; i++) {
				Move m = list[i].move;
				const auto [moveFrom, moveTo] = m.Unpack();
				const auto pt = bb.GetPieceAtIndex(moveFrom);

				if (m == ttMove) {
					list[i].Score = TT_SCORE;
				}
				else if (m == ss->KillerMove) {
					list[i].Score = KILLER_SCORE;
				}
				else if (bb.GetPieceAtIndex(moveTo) != Piece::NONE && !m.IsCastle()) {
					const auto capturedPiece = bb.GetPieceAtIndex(moveTo);
					auto& hist = history.CaptureHistory[pc][pt][moveTo][capturedPiece];
					list[i].Score = (MVVMult * GetPieceValue(capturedPiece)) + hist;
				}
				else {
					i32 contIdx = MakePiece(pc, pt);

					list[i].Score  = 2 * history.MainHistory[pc][m.GetMoveMask()];
					list[i].Score += 2 * (*(ss - 1)->ContinuationHistory)[contIdx][moveTo];
					list[i].Score +=	 (*(ss - 2)->ContinuationHistory)[contIdx][moveTo];
					list[i].Score +=	 (*(ss - 4)->ContinuationHistory)[contIdx][moveTo];
					list[i].Score +=	 (*(ss - 6)->ContinuationHistory)[contIdx][moveTo];

					if (ss->Ply < LowPlyCount) {
						list[i].Score += ((2 * LowPlyCount + 1) * history.PlyHistory[ss->Ply][m.GetMoveMask()]) / (2 * ss->Ply + 1);
					}

					if ((pos.State->CheckSquares[pt] & SquareBB(moveTo)) != 0) {
						list[i].Score += CheckBonus;
					}
				}

				if (pt == HORSIE) {
					list[i].Score += 200;
				}
			}
		}


		inline i32 FindNext() {
			auto best = moveIndex;
			auto bestScore = list[moveIndex].Score;

			for (auto i = moveIndex + 1; i < listSize; ++i) {
				if (list[i].Score > bestScore) {
					best = i;
					bestScore = list[i].Score;
				}
			}

			if (best != moveIndex)
				std::swap(list[moveIndex], list[best]);

			return moveIndex++;
		}


		inline Move SelectNext(auto predicate) {
			while (moveIndex < listSize) {
				const auto idx = FindNext();
				const auto move = list[idx].move;

				if (predicate(move))
					return move;
			}

			return Move::Null();
		}
	};

}



#endif // !MOVEPICKER_H
