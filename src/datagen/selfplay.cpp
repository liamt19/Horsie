

#include "selfplay.h"

#include "../movegen.h"
#include "../position.h"
#include "../threadpool.h"
#include "../util/timer.h"

#include <cassert>
#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <random>

#if defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif
#include <cstring>

namespace Horsie::Datagen {

    inline void ClearConsoleLines(i32 nLines) {
        
    }

    inline void HideCursor() {

    }
        
    std::atomic<i32> seed(static_cast<i32>(std::chrono::steady_clock::now().time_since_epoch().count()));
        
    thread_local std::mt19937 rng([]{ return std::mt19937(seed.fetch_add(1, std::memory_order_relaxed)); }());

    static i32 RandNext(i32 min, i32 max) {
        std::uniform_int_distribution<> distr(min, max - 1);
        return distr(rng);
    }


    void MonitorDG(u64 nodes, u64 depth, u64 games, u64 threads, bool dfrc) {
        
    }

    void GetStartPos(SearchThread* thread, Position& pos, SearchLimits& prelimInfo, bool dfrc) {
        
    }

    void RunGames(i32 threadID, u64 softNodeLimit, u64 depthLimit, u64 gamesToRun, bool dfrc) {
        
    }

    void AddResultsAndWrite(std::vector<BulletFormatEntry> datapoints, GameResult gr, std::ofstream& outputWriter) {
        
    }

    void DGSetupThread(Position& pos, SearchThread& td) {
        
    }
    

}