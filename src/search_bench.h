#pragma once

#ifndef SEARCH_BENCH_H
#define SEARCH_BENCH_H


#include <chrono>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <list>
#include <thread>
#include <barrier>

#include "threadpool.h"
#include "position.h"
#include "util.h"
#include "search.h"

using namespace Horsie::Search;


namespace Horsie {
    
    void DoBench(SearchThreadPool& SearchPool, i32 depth = 12, bool openBench = false) {

        Position pos = Position(InitialFEN);
        SearchThread* thread = SearchPool.MainThread();

        auto odf = thread->OnDepthFinish;
        auto osf = thread->OnSearchFinish;
        thread->OnDepthFinish = []() {};
        thread->OnSearchFinish = []() {};

        SearchLimits limits;
        limits.MaxDepth = depth;
        limits.MaxSearchTime = INT32_MAX;

        u64 totalNodes = 0;

        SearchPool.TTable.Clear();
        SearchPool.Clear();

        auto before = std::chrono::system_clock::now();
        for (std::string fen : BenchFENs)
        {
            pos.LoadFromFEN(fen);

            SearchPool.StartSearch(pos, limits);
            SearchPool.WaitForMain();

            u64 thisNodeCount = SearchPool.GetNodeCount();
            totalNodes += thisNodeCount;

            if (!openBench) {
                std::cout << std::left << std::setw(76) << fen << "\t" << std::to_string(thisNodeCount) << std::endl;
            }

            SearchPool.TTable.Clear();
            SearchPool.Clear();
        }

        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - before);

        auto dur = duration.count();
        auto durSeconds = duration.count() / 1000;
        auto durMillis = duration.count() % 1000;
        auto nps = (u64)(totalNodes / ((double)dur / 1000));

        if (openBench) {
            std::cout << "info string " << durSeconds << "." << durMillis << " seconds" << std::endl;
            std::cout << totalNodes << " nodes " << nps << " nps" << std::endl;
        }
        else {
            std::cout << std::endl << "Nodes searched: " << totalNodes << " in " << durSeconds << "." << durMillis << " s (" << FormatWithCommas(nps) << " nps)" << std::endl;
        }

        thread->OnDepthFinish = odf;
        thread->OnSearchFinish = osf;
    }

}


#endif