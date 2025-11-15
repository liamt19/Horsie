
#include "tt.h"

#include "util/alloc.h"

#include <thread>
#include <vector>

namespace Horsie {

    bool TranspositionTable::Probe(u64 hash, TTEntry*& tte) const {
        TTCluster* const cluster = GetCluster(hash);
        tte = (TTEntry*)cluster;

        auto key = static_cast<u16>(hash);

        for (i32 i = 0; i < TTCluster::EntriesPerCluster; i++) {
            //  If the entry's key matches, or the entry is empty, then pick this one.
            if (tte[i].Key == key || tte[i].IsEmpty()) {
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
        for (i32 i = 1; i < TTCluster::EntriesPerCluster; i++) {
            if (replace->Quality(Age) > tte[i].Quality(Age)) {
                replace = &tte[i];
            }
        }

        tte = replace;
        return false;
    }

    void TranspositionTable::Initialize(i32 mb) {
        if (Clusters) {
            AlignedFree(Clusters);
        }

        u64 size = u64(mb) * 1024 * 1024 / sizeof(TTCluster);
        ClusterCount = size;
        Clusters = AlignedAlloc<TTCluster>(ClusterCount);

        Clear();
    }

    void TranspositionTable::Clear() {

        const auto numThreads = Horsie::Threads.CurrentValue;
        const u64 clustersPerThread = (ClusterCount / static_cast<u64>(numThreads));
        std::vector<std::thread> threads{};

        for (i32 i = 0; i < numThreads; ++i) {
            threads.emplace_back([this, clustersPerThread, numThreads, i] {
                const u64 start = clustersPerThread * static_cast<u64>(i);
                const u64 length = (i == numThreads - 1) ? ClusterCount - start : clustersPerThread;

                std::memset(&Clusters[start], 0, sizeof(TTCluster) * length);
            });
        }

        for (auto& thread : threads)
            thread.join();

        Age = 0;
    }

    u32 TranspositionTable::Hashfull() const {
        u32 entries = 0;
        for (size_t i = 0; i < 1000; i++) {
            const auto& cluster = Clusters[i];

            for (size_t j = 0; j < TTCluster::EntriesPerCluster; j++) {
                const auto e = cluster.entries[j];

                if (!e.IsEmpty() && e.Age() == Age)
                    entries++;
            }
        }

        return entries / TTCluster::EntriesPerCluster;
    }

    void TTEntry::Update(u64 key, i16 score, TTNodeType nodeType, i32 depth, Move move, i16 statEval, u8 age, bool isPV) {
        u16 k = static_cast<u16>(key);

        if (move != Move::Null() || k != Key) {
            BestMove = move;
        }

        if (nodeType == TTNodeType::Exact
            || k != Key
            || age != Age()
            || depth + (2 * isPV) > Depth() - 4) {
            
            Key = k;
            SetScore(score);
            SetStatEval(statEval);
            _depth = static_cast<u8>(depth - DepthOffset);
            _AgePVType = static_cast<u8>(age | ((isPV ? 1u : 0u) << 2) | static_cast<u32>(nodeType));
        }
    }
}
