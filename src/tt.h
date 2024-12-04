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


constexpr i32 TT_BOUND_MASK = 0x3;
constexpr i32 TT_PV_MASK = 0x4;
constexpr i32 TT_AGE_MASK = 0xF8;
constexpr i32 TT_AGE_INC = 0x8;
constexpr i32 TT_AGE_CYCLE = 255 + TT_AGE_INC;

constexpr i32 MinTTClusters = 1000;

constexpr i32 EntriesPerCluster = 3;





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
        i16 _Score;           //  16 bits
        i16 _StatEval;		//  16 bits
        Move BestMove;          //  16 bits
        u16 Key;             //  16 bits
        u8 _AgePVType;        //  5 + 2 + 1 bits
        u8 _depth;            //  8 bits


        constexpr i16 Score() const { return _Score; }
        constexpr void SetScore(i16 n) { _Score = n; }

        constexpr i16 StatEval() const { return _StatEval; }
        constexpr void SetStatEval(i16 n) { _StatEval = n; }

        constexpr i32 Age() const { return _AgePVType & TT_AGE_MASK; }
        constexpr bool PV() const { return (_AgePVType & TT_PV_MASK) != 0; }
        constexpr i32 Bound() const { return _AgePVType & TT_BOUND_MASK; }

        constexpr i32 Depth() const { return (i32)(_depth + DepthOffset); }
        constexpr void SetDepth(i32 n) { _depth = (u8)(n - DepthOffset); }
        constexpr i32 RawDepth() const { return _depth; }

        constexpr bool IsEmpty() const { return _depth == 0; }

        constexpr i8 RelAge(u8 age) const { return (i8)((TT_AGE_CYCLE + age - _AgePVType) & TT_AGE_MASK); }

        void Update(u64 key, i16 score, TTNodeType nodeType, i32 depth, Move move, i16 statEval, bool isPV = false);

        static constexpr i32 DepthNone = -6;

    private:
        static constexpr i32 KeyShift = 64 - (sizeof(u16) * 8);
        static constexpr i32 DepthOffset = -7;
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
        void Initialize(i32 mb);
        bool Probe(u64 hash, TTEntry*& tte) const;

        void TTUpdate() {
            Age += TT_AGE_INC;
        }

        void Clear() const {
            std::memset(Clusters, 0, sizeof(TTCluster) * ClusterCount);
        }

        TTCluster* GetCluster(u64 hash) const
        {
            return Clusters + (u64)(((uint128_t(hash) * uint128_t(ClusterCount)) >> 64));
        }

        TTCluster* Clusters = nullptr;
        u16 Age = 0;
        u64 ClusterCount = 0;

        static const i32 DefaultTTSize = 32;
        static const i32 MaxSize = 1048576;
    };

    extern TranspositionTable TT;

}

#endif

