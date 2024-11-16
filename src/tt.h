#pragma once

#ifndef TT_H
#define TT_H

#include "types.h"
#include "move.h"

#include <cstring>



#ifdef _MSC_VER
#include <__msvc_int128.hpp>
using uint128_t = std::_Unsigned128;
#else
using uint128_t = unsigned __int128;
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
        short _Score;           //  16 bits
        short _StatEval;		//  16 bits
        Move BestMove;          //  16 bits
        ushort Key;             //  16 bits
        byte _AgePVType;        //  5 + 2 + 1 bits
        byte _depth;            //  8 bits


        constexpr short Score() const { return _Score; }
        constexpr void SetScore(short n) { _Score = n; }

        constexpr short StatEval() const { return _StatEval; }
        constexpr void SetStatEval(short n) { _StatEval = n; }

        constexpr int Age() const { return _AgePVType & TT_AGE_MASK; }
        constexpr bool PV() const { return (_AgePVType & TT_PV_MASK) != 0; }
        constexpr int Bound() const { return _AgePVType & TT_BOUND_MASK; }

        constexpr int Depth() const { return (int)(_depth + DepthOffset); }
        constexpr void SetDepth(int n) { _depth = (byte)(n - DepthOffset); }
        constexpr int RawDepth() const { return _depth; }

        constexpr bool IsEmpty() const { return _depth == 0; }

        constexpr sbyte RelAge(byte age) const { return (sbyte)((TT_AGE_CYCLE + age - _AgePVType) & TT_AGE_MASK); }

        void Update(ulong key, short score, TTNodeType nodeType, int depth, Move move, short statEval, bool isPV = false);

        static constexpr int DepthNone = -6;

    private:
        static constexpr int KeyShift = 64 - (sizeof(ushort) * 8);
        static constexpr int DepthOffset = -7;
    };


    struct alignas(32) TTCluster {
        TTEntry entries[3];
        char _pad[2];

        void Clear() {
            std::memset(&entries[0], 0, sizeof(TTEntry) * 3);
        }
    };

    static_assert(sizeof(TTCluster) == 32, "Unexpected Cluster size");



    class TranspositionTable {
    public:
        void Initialize(int mb);
        bool Probe(ulong hash, TTEntry*& tte) const;

        void TTUpdate() {
            Age += TT_AGE_INC;
        }

        void Clear() const {
            std::memset(Clusters, 0, sizeof(TTCluster) * ClusterCount);
        }

        TTCluster* GetCluster(ulong hash) const
        {
            return Clusters + (ulong)(((uint128_t(hash) * uint128_t(ClusterCount)) >> 64));
        }

        TTCluster* Clusters = nullptr;
        ushort Age = 0;
        ulong ClusterCount = 0;

        static const int DefaultTTSize = 32;
        static const int MaxSize = 1048576;
    };

    extern TranspositionTable TT;

}

#endif

