#pragma once

#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <thread>
#include <barrier>

#include <array>
#include <vector>
#include <chrono>
#include <algorithm>
#include <functional>
#include <mutex>

#include "tt.h"
#include "move.h"
#include "history.h"
#include "search.h"
#include "position.h"
#include "util/alloc.h"
#include "search_options.h"

using namespace Horsie::Util::ThingsIStoleFromStormphrax;
using namespace Horsie::Search;

namespace Horsie {

    class SearchThreadPool;
    class SearchThread;

    class Thread {
    public:
        Thread(i32 n);
        virtual ~Thread();

        void   idle_loop();
        void   start_searching();
        void   WaitForThreadFinished();
        size_t id() const { return idx; }

        std::unique_ptr<SearchThread> worker;

    private:
        std::mutex mutex;
        std::condition_variable cv;
        size_t idx;
        size_t nthreads;
        bool exit = false;
        bool searching = true;
        std::thread stdThread;
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
        i32 CheckupCount{};
        bool Searching{};
        bool Quit{};
        constexpr bool IsMain() const { return ThreadIdx == 0; }
        SearchThreadPool* AssocPool;
        TranspositionTable* TT;

        std::chrono::system_clock::time_point StartTime{};
        i32 TimeLimit{};
        u64 MaxNodes{};

        std::function<void()> OnDepthFinish;
        std::function<void()> OnSearchFinish;
        std::vector<RootMove> RootMoves{};
        Position RootPosition;
        HistoryTable History{};

        i32 SoftTimeLimit{};
        Util::NDArray<u64, 64, 64> NodeTable{};

        //SearchThread(i32 idx = 0) : ThreadIdx(idx) {

        //}

        void MainThreadSearch();

        void Search(SearchLimits& info);

        template <SearchNodeType NodeType>
        i32 Negamax(Position& pos, SearchStackEntry* ss, i32 alpha, i32 beta, i32 depth, bool cutNode);

        template <SearchNodeType NodeType>
        i32 QSearch(Position& pos, SearchStackEntry* ss, i32 alpha, i32 beta);

        void UpdateCorrectionHistory(Position& pos, i32 diff, i32 depth);
        i16 AdjustEval(Position& pos, i32 us, i16 rawEval);

        void AssignProbCutScores(Position& pos, ScoredMove* list, i32 size);
        void AssignQuiescenceScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, i32 size, Move ttMove);
        void AssignScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, i32 size, Move ttMove);
        Move OrderNextMove(ScoredMove* moves, i32 size, i32 listIndex);

        void UpdatePV(Move* pv, Move move, Move* childPV);

        void UpdateContinuations(SearchStackEntry* ss, i32 pc, i32 pt, i32 sq, i32 bonus);
        void UpdateStats(Position& pos, SearchStackEntry* ss, Move bestMove, i32 bestScore, i32 beta, i32 depth, Move* quietMoves, i32 quietCount, Move* captureMoves, i32 captureCount);

        std::string Debug_GetMovesPlayed(SearchStackEntry* ss);



        i64 GetSearchTime() const {
            auto now = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - StartTime);
            return duration.count();
        }

        void MakeMoveTime(SearchLimits& limits) {
            limits.MaxSearchTime = std::min(limits.Increment + std::max(limits.PlayerTime / 2, limits.PlayerTime / limits.MovesToGo), limits.PlayerTime);

            //  Values from Clarity
            SoftTimeLimit = static_cast<i32>(0.65 * ((static_cast<double>(limits.PlayerTime) / limits.MovesToGo) + (limits.Increment * 3 / 4.0)));
        }

        void Reset() {
            Nodes = 0;
            PVIndex = RootDepth = SelDepth = CompletedDepth = CheckupCount = NMPPly = 0;
        }

        constexpr bool HasSoftTime() const { return SoftTimeLimit != 0; }
        Move CurrentMove() const { return RootMoves[PVIndex].move; }
        bool CheckTime() const { return (GetSearchTime() > TimeLimit - SEARCH_TIME_BUFFER); }

        inline i32 GetRFPMargin(i32 depth, bool improving) const { return (depth - (improving)) * RFPMargin; }
        inline i32 StatBonus(i32 depth) const { return std::min((StatBonusMult * depth) - StatBonusSub, StatBonusMax); }
        inline i32 StatMalus(i32 depth) const { return std::min((StatMalusMult * depth) - StatMalusSub, StatMalusMax); }

    private:
        const i32 CheckupMax = 512;
    };


	class SearchThreadPool {
	public:
        SearchLimits SharedInfo;
		std::vector<Thread*> Threads;
        TranspositionTable TTable;
        std::atomic_bool StopThreads{};
        std::barrier<> Blocker{ 2 };

        SearchThreadPool(i32 n = 1) {
            TTable.Initialize(Horsie::Hash);
            Resize(n);
        }

        constexpr SearchThread* MainThread() const { return Threads.front()->worker.get(); }
        constexpr Thread* MainThreadBase() const { return Threads.front(); }

        void Resize(int newThreadCount);
        void StartSearch(Position& rootPosition, const SearchLimits& rootInfo);
        void StartSearch(Position& rootPosition, const SearchLimits& rootInfo, ThreadSetup& setup);
        SearchThread* GetBestThread() const;
        void StartThreads() const;
        void WaitForSearchFinished() const;
        void BlockCallerUntilFinished();
        void Clear() const;

        u64 GetNodeCount() const {
            u64 sum = 0;
            for (auto& td : Threads) {
                sum += td->worker.get()->Nodes;
            }
            return sum;
        }

        static const i32 MaxThreads = 512;
	};

}

#endif // !THREADPOOL_H
