#include "tt.h"


namespace Horsie {

    TranspositionTable TT;

    bool TranspositionTable::Probe(ulong hash, TTEntry* tte)
    {
        TTCluster* cluster = GetCluster(hash);
        tte = (TTEntry*)cluster;

        auto key = (ushort)hash;

        for (int i = 0; i < EntriesPerCluster; i++)
        {
            //  If the entry's key matches, or the entry is empty, then pick this one.
            if (tte[i].Key == key || tte[i].IsEmpty())
            {
                tte[i]._AgePVType = (sbyte)(TT.Age | (tte[i]._AgePVType & (TT_AGE_INC - 1)));

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
            if (((replace->_depth - (TT_AGE_CYCLE + Age - replace->_AgePVType)) & TT_AGE_MASK) >
                ((tte[i]._depth - (TT_AGE_CYCLE + Age - tte[i]._AgePVType)) & TT_AGE_MASK))
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

    }

    void TTEntry::Update(ulong key, short score, TTNodeType nodeType, int depth, Move move, short statEval, bool isPV) {
        
        //if (!move.IsNull() || (ushort)key != this.Key)
        if (move != Move::Null() || (ushort)key != Key)
        {
            BestMove = move;
        }

        if (nodeType == TTNodeType::Exact
            || (ushort)key != Key
            || depth + (isPV ? 2 : 0) > _depth - 11)
        {
            Key = (ushort)key;
            SetScore(score);
            SetStatEval(statEval);
            SetDepth((sbyte)depth);
            _AgePVType = (sbyte)(TT.Age | ((isPV ? 1 : 0) << 2) | (int)nodeType);
        }
    }

}