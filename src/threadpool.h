#pragma once

/*

Copied from https://github.com/liamt19/Lizard/blob/main/Logic/Threads/SearchThreadPool.cs:

Some of the thread logic in this class is based on Stockfish's Thread class
(StartThreads, WaitForSearchFinished, and the general concepts in StartSearch), the sources of which are here:
https://github.com/official-stockfish/Stockfish/blob/master/src/thread.cpp
https://github.com/official-stockfish/Stockfish/blob/master/src/thread.h

*/

#include "defs.h"
#include "history.h"
#include "move.h"
#include "position.h"
#include "search.h"
#include "search_options.h"
#include "tt.h"
#include "util/NDArray.h"
#include "util/timer.h"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

using namespace Horsie::Util::ThingsIStoleFromStormphrax;
using namespace Horsie::Search;

namespace Horsie {

    class SearchThreadPool;
    class SearchThread;

    class Thread {
    public:
        Thread(i32 n);
        virtual ~Thread();

        void IdleLoop();
        void WakeUp();
        void WaitForThreadFinished();

        std::unique_ptr<SearchThread> Worker;

    private:
        std::mutex Mut;
        std::condition_variable CondVar;
        std::thread SysThread;
        bool Quit = false;
        bool Active = true;
    };

    class SearchThread {
    public:
        u64 Nodes{};
        i32 NMPPly{};
        i32 ThreadIdx{};
        i32 PVIndex{};
        i32 RootDepth{};
        i32 SelDepth{};
        i32 CompletedDepth{};

        i32 HardTimeLimit{};
        i32 SoftTimeLimit{};
        u64 HardNodeLimit{};

        std::atomic_bool StopSearching{};
        bool IsDatagen{};

        SearchThreadPool* AssocPool{};
        TranspositionTable* TT{};
        Position RootPosition;
        HistoryTable History{};

        Timepoint StartTime{};

        std::function<void()> OnDepthFinish;
        std::function<void()> OnSearchFinish;
        std::vector<RootMove> RootMoves{};

        constexpr bool HasSoftTime() const { return SoftTimeLimit != 0; }
        constexpr bool IsMain() const { return ThreadIdx == 0; }

        void MainThreadSearch();

        void Search(SearchLimits& info);

        template <SearchNodeType NodeType>
        i32 Negamax(Position& pos, SearchStackEntry* ss, i32 alpha, i32 beta, i32 depth, bool cutNode);

        template <SearchNodeType NodeType>
        i32 QSearch(Position& pos, SearchStackEntry* ss, i32 alpha, i32 beta);

        void UpdateCorrectionHistory(Position& pos, i32 diff, i32 depth);
        i16 AdjustEval(Position& pos, i32 us, i16 rawEval) const;

        void AssignProbcutScores(Position& pos, ScoredMove* list, i32 size) const;
        void AssignQuiescenceScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, i32 size, Move ttMove) const;
        void AssignScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, i32 size, Move ttMove) const;
        Move OrderNextMove(ScoredMove* moves, i32 size, i32 listIndex) const;

        void UpdatePV(Move* pv, Move move, Move* childPV) const;

        void UpdateContinuations(SearchStackEntry* ss, i32 pc, i32 pt, i32 sq, i32 bonus) const;
        void UpdateStats(Position& pos, SearchStackEntry* ss, Move bestMove, i32 bestScore, i32 beta, i32 depth, Move* quietMoves, i32 quietCount, Move* captureMoves, i32 captureCount);

        std::string Debug_GetMovesPlayed(SearchStackEntry* ss) const;

        void PrintSearchInfo() const;

        bool ShouldStop() const;
        void SetStop(bool flag = true);
        void CheckLimits();

        void Reset() {
            Nodes = 0;
            PVIndex = RootDepth = SelDepth = CompletedDepth = NMPPly = 0;
        }

        Move CurrentMove() const { return RootMoves[PVIndex].move; }
        auto GetSearchTime() const { return Timepoint::TimeSince(StartTime); }
        bool HardTimeReached() const { return (GetSearchTime() > HardTimeLimit - MoveOverhead); }

        inline i32 GetRFPMargin(i32 depth, bool improving) const { return (depth - (improving)) * RFPMargin; }
        inline i32 StatBonus(i32 depth) const { return std::min((i32)(StatBonusMult * depth) - StatBonusSub, (i32)StatBonusMax); }
        inline i32 StatMalus(i32 depth) const { return std::min((i32)(StatMalusMult * depth) - StatMalusSub, (i32)StatMalusMax); }

    private:
        const u32 CheckupFrequency = 1023;
    };

    class SearchThreadPool {
    public:
        SearchLimits SharedInfo;
        std::vector<Thread*> Threads;
        TranspositionTable TTable;

        SearchThreadPool(i32 n = 1) {
            TTable.Initialize(Horsie::Hash);
            Resize(n);
        }

        constexpr SearchThread* MainThread() const { return Threads.front()->Worker.get(); }
        constexpr Thread* MainThreadBase() const { return Threads.front(); }

        void StopAllThreads() { 
            for (i32 i = 1; i < Threads.size(); i++)
                Threads[i]->Worker->SetStop(true);

            MainThread()->SetStop(true);
        }

        void StartAllThreads() {
            for (i32 i = 1; i < Threads.size(); i++)
                Threads[i]->Worker->SetStop(false);

            MainThread()->SetStop(false);
        }

        void Resize(i32 newThreadCount);
        void StartSearch(Position& rootPosition, const SearchLimits& rootInfo);
        void StartSearch(Position& rootPosition, const SearchLimits& rootInfo, ThreadSetup& setup);
        void WaitForMain() const;
        SearchThread* GetBestThread() const;
        void SendBestMove() const;
        void AwakenHelperThreads() const;
        void WaitForSearchFinished() const;
        void Clear() const;

        u64 GetNodeCount() const {
            u64 sum = 0;
            for (auto& td : Threads) {
                sum += td->Worker.get()->Nodes;
            }
            return sum;
        }
    };
}
