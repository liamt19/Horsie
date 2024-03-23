
#include "search.h"
#include "position.h"



using namespace Horsie::Search;

namespace Horsie {

	template <SearchNodeType NodeType>
	int Negamax(Position& pos, SearchStackEntry* ss, int alpha, int beta, int depth, bool cutNode) {
		constexpr bool isRoot = NodeType == SearchNodeType::RootNode;
		constexpr bool isPV = NodeType != SearchNodeType::NonPVNode;

		if (depth <= 0)
		{
			return QSearch<NodeType>(pos, ss, alpha, beta, depth);
		}

		Bitboard bb = pos.bb;
		//SearchThread thisThread = pos.Owner;
		SearchThread thisThread = SearchThread();

		ulong posHash = pos.State->Hash;

		Move bestMove = Move.Null;

		int ourColor = (int) pos.ToMove;
		int score = -ScoreMate - MaxPly;
		int bestScore = -ScoreInfinite;

		int startingAlpha = alpha;

		short eval = ss->StaticEval;

		bool doSkip = ss->Skip != Move::Null();
		bool improving = false;




		return 1;
	}


	template <SearchNodeType NodeType>
	int QSearch(Position& pos, SearchStackEntry* ss, int alpha, int beta, int depth, bool cutNode) {

		return 1;
	}
}