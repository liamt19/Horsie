#include "tt.h"


namespace Horsie {

    TranspositionTable TT;

    bool TranspositionTable::Probe(ulong hash, TTEntry* tte) const
    {
        TTCluster* const cluster = GetCluster(hash);
        tte = (TTEntry*)cluster;

        auto key = (ushort)hash;

        for (int i = 0; i < EntriesPerCluster; i++)
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
        for (int i = 1; i < EntriesPerCluster; i++)
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



    void TranspositionTable::Initialize(int mb)
    {
        ulong size = ulong(mb) * 1024 * 1024 / sizeof(TTCluster);
        Clusters = new TTCluster[size];
        ClusterCount = size;
        std::memset(Clusters, 0, sizeof(TTCluster) * size);
    }

    void TTEntry::Update(ulong key, short score, TTNodeType nodeType, int depth, Move move, short statEval, bool isPV) {

        ushort k = (ushort)key;

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
            _depth = (byte)(depth - DepthOffset);
            _AgePVType = (byte)(TT.Age | ((isPV ? 1 : 0) << 2) | (uint)nodeType);
        }
    }

}