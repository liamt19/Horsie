

#include "selfplay.h"

#include "../movegen.h"
#include "../position.h"
#include "../threadpool.h"
#include "../util/timer.h"

#include <cassert>
#include <chrono>
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

    void MonitorDG(u64 nodes, u64 depth, u64 games, u64 threads, bool dfrc) {

    }

    void GetStartPos(SearchThread* thread, Position& pos, SearchLimits& prelimInfo, bool dfrc) {

    }

    void RunGames(i32 threadID, u64 softNodeLimit, u64 depthLimit, u64 gamesToRun, bool dfrc) {

    }

    void AddResultsAndWrite(std::vector<BulletFormatEntry> datapoints, GameResult gr, std::ofstream& outputWriter) {
        for (i32 i = 0; i < datapoints.size(); i++) {
            datapoints[i].SetResult(gr);
            outputWriter << datapoints[i];
        }

        outputWriter.flush();
    }

    void DGSetupThread(Position& pos, SearchThread& td) {
        td.Reset();

        ScoredMove rms[MoveListSize] = {};
        i32 size = Generate<GenLegal>(pos, &rms[0], 0);

        td.RootMoves.clear();
        td.RootMoves.shrink_to_fit();
        td.RootMoves.reserve(size);

        for (i32 j = 0; j < size; j++) {
            td.RootMoves.push_back(RootMove(rms[j].move));
        }

        td.RootPosition.LoadFromFEN(pos.GetFEN());
    }
    

}