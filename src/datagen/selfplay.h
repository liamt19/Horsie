#pragma once

#include "../defs.h"
#include "../position.h"
#include "../threadpool.h"
#include "bullet_format.h"

#include <vector>


namespace Horsie::Datagen {

    constexpr auto HashSize = 8;

    constexpr auto MinOpeningPly = 8;
    constexpr auto MaxOpeningPly = 9;

    constexpr auto SoftNodeLimit = 5000;

    constexpr auto DepthLimit = 14;

    constexpr auto WritableDataLimit = 512;

    constexpr auto AdjudicateMoves = 4;
    constexpr auto AdjudicateScore = 3000;

    constexpr auto MaxFilteringScore = 6000;
    constexpr auto MaxOpeningScore = 1200;

    constexpr i32 DefaultThreadCount = 1;
    constexpr i32 DefaultGameCount = INT32_MAX;

    inline std::vector<u64> gameCounts{};
    inline std::vector<u64> positionCounts{};
    inline std::vector<double> threadNPS{};
    inline std::vector<double> threadDepths{};
    inline std::vector<SearchThread*> threadRefs{};

    struct BulletFormatEntry;

    void MonitorDG(u64 nodes, u64 depth, u64 games, u64 threads, bool dfrc);
    void DGSetupThread(Position& pos, SearchThread& td);
    void RunGames(i32 threadID, u64 softNodeLimit, u64 depthLimit, u64 gamesToRun, bool dfrc);

    void AddResultsAndWrite(std::vector<BulletFormatEntry> datapoints, GameResult gr, std::ofstream& outputWriter);

}
