#pragma once

#ifndef TT_H
#define TT_H

#include "types.h"
#include "move.h"


#ifdef _MSC_VER
#include <__msvc_int128.hpp>
using uint128_t = std::_Unsigned128;
#endif


constexpr int TT_BOUND_MASK = 0x3;
constexpr int TT_PV_MASK = 0x4;
constexpr int TT_AGE_MASK = 0xF8;
constexpr int TT_AGE_INC = 0x8;
constexpr int TT_AGE_CYCLE = 255 + TT_AGE_INC;

constexpr int MinTTClusters = 1000;

constexpr int EntriesPerCluster = 3;




namespace Horsie {

	enum class TTNodeType
	{
		Invalid,
		/// Upper Bound
		Beta,
		/// Lower bound
		Alpha,
		Exact = Beta | Alpha
	};

	struct TTEntry {
		int _ScoreStatEval;
		Move BestMove;
		ushort Key;
		sbyte _AgePVType;
		sbyte _depth;


		constexpr short Score() const { return (short)(_ScoreStatEval & 0xFFFF); }
		constexpr void SetScore(short n) { _ScoreStatEval = (_ScoreStatEval & ~0xFFFF) | n; }

		constexpr short StatEval() const { return (short)((_ScoreStatEval & 0xFFFF0000) >> 16); }
		constexpr void SetStatEval(short n) { _ScoreStatEval = (int)((_ScoreStatEval & 0x0000FFFF) | (n << 16)); }

		constexpr int Age() const { return _AgePVType & TT_AGE_MASK; }
		constexpr bool PV() const { return (_AgePVType & TT_PV_MASK) != 0; }
		constexpr int Bound() const { return _AgePVType & TT_BOUND_MASK; }

		constexpr sbyte Depth() const { return (_depth - DepthOffset); }
		constexpr void SetDepth(int n) { _depth = (sbyte)(n + DepthOffset); }

		constexpr bool IsEmpty() const { return _depth == 0; }

		void Update(ulong key, short score, TTNodeType nodeType, int depth, Move move, short statEval, bool isPV = false);

	private:
		const int KeyShift = 64 - (sizeof(ushort) * 8);
		const int DepthOffset = 7;
		const int DepthNone = -6;
	};


	struct TTCluster {
		TTEntry entries[3];
		char _pad[2];

		void Clear() {
			std::memset(&entries[0], 0, sizeof(TTEntry) * 3);
		}
	};



	class TranspositionTable {
	public:
		void Initialize(int mb);
		bool Probe(ulong hash, TTEntry* tte);

		void TTUpdate() {
			Age += TT_AGE_INC;
		}

		void Clear() {
			for (ulong i = 0; i < ClusterCount; i++)
			{
				TTEntry* cluster = (TTEntry*)&Clusters[i];

				Clusters[i].Clear();
			}
		}

		TTCluster* GetCluster(ulong hash)
		{
			return Clusters + ((ulong)((uint128_t(hash) * uint128_t(ClusterCount)) >> 64));
		}

		TTCluster* Clusters = nullptr;
		ushort Age = 0;
		ulong ClusterCount = 0;
	};

	extern TranspositionTable TT;

}

#endif

