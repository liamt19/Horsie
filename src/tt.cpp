#include "tt.h"
#include "util/alloc.h"

namespace Horsie {

    TranspositionTable TT;

    bool TranspositionTable::Probe(u64 hash, TTEntry*& tte) const
    {
        TTCluster* const cluster = GetCluster(hash);
        tte = (TTEntry*)cluster;

        auto key = static_cast<u16>(hash);

        for (i32 i = 0; i < EntriesPerCluster; i++)
        {
            //  If the entry's key matches, or the entry is empty, then pick this one.
            if (tte[i].Key == key || tte[i].IsEmpty())
            {
                tte = &tte[i];

                //  We return true if the entry isn't empty, which means that tte is valid.
                //  Check tte[0] here, not tte[i].
                return !tte[0].IsEmpty();
            }
        }

        //  We didn't find an entry for this hash, so instead we will choose one of the 
        //  non-working entries in this cluster to possibly be overwritten / updated, and return false.

        //  Replace the first entry, unless the 2nd or 3rd is a better option.
        TTEntry* replace = tte;
        for (i32 i = 1; i < EntriesPerCluster; i++)
        {
            if ((replace->RawDepth() - replace->RelAge(Age)) >
                (  tte[i].RawDepth() -   tte[i].RelAge(Age)))
            {
                replace = &tte[i];
            }
        }

        tte = replace;
        return false;
    }



    void TranspositionTable::Initialize(i32 mb)
    {
        if (Clusters) {
            AlignedFree(Clusters);
        }

        u64 size = u64(mb) * 1024 * 1024 / sizeof(TTCluster);
        ClusterCount = size;
        Clusters = AlignedAlloc<TTCluster>(ClusterCount);
        
        std::memset(Clusters, 0, sizeof(TTCluster) * size);
    }

    void TTEntry::Update(u64 key, i16 score, TTNodeType nodeType, i32 depth, Move move, i16 statEval, bool isPV) {

        u16 k = static_cast<u16>(key);

        if (move != Move::Null() || k != Key)
        {
            BestMove = move;
        }

        if (nodeType == TTNodeType::Exact
            || k != Key
            || depth + (isPV ? 2 : 0) > _depth - 4 + DepthOffset)
        {
            Key = k;
            SetScore(score);
            SetStatEval(statEval);
            _depth = static_cast<u8>(depth - DepthOffset);
            _AgePVType = static_cast<u8>(TT.Age | ((isPV ? 1 : 0) << 2) | static_cast<u32>(nodeType));
        }
    }

}