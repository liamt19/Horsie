
#include "threadpool.h"

#include "defs.h"
#include "move.h"
#include "movegen.h"
#include "types.h"
#include "util.h"

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

/*

Copied from https://github.com/liamt19/Lizard/blob/main/Logic/Threads/SearchThreadPool.cs:

Some of the thread logic in this class is based on Stockfish's Thread class
(StartThreads, WaitForSearchFinished, and the general concepts in StartSearch), the sources of which are here:
https://github.com/official-stockfish/Stockfish/blob/master/src/thread.cpp
https://github.com/official-stockfish/Stockfish/blob/master/src/thread.h

*/


namespace Horsie {

    void SearchThreadPool::Resize(i32 newThreadCount) {
        if (Threads.size() > 0) {
            WaitForMain();

            while (Threads.size() > 0)
                delete Threads.back(), Threads.pop_back();
        }

        for (i32 i = 0; i < newThreadCount; i++) {
            auto td = new Thread(i);
            auto worker = td->worker.get();
            worker->ThreadIdx = i;
            worker->AssocPool = this;
            worker->TT = &TTable;

            worker->OnDepthFinish = [worker]() { worker->PrintSearchInfo(); };
            worker->OnSearchFinish = [&]() { SendBestMove(); };

            Threads.push_back(td);
        }

        PoolHistory.clear();
        PoolHistory.resize(newThreadCount);
        for (auto& v : PoolHistory) {
            v.reserve(MaxDepth);
        }

        WaitForMain();
    }

    void SearchThreadPool::StartSearch(Position& rootPosition, const SearchLimits& rootInfo) {
        ThreadSetup setup{};
        StartSearch(rootPosition, rootInfo, setup);
    }

    void SearchThreadPool::StartSearch(Position& rootPosition, const SearchLimits& rootInfo, ThreadSetup& setup) {
        WaitForMain();
        MainThread()->StartTime = std::chrono::system_clock::now();

        StartAllThreads();
        SharedInfo = rootInfo;
        SharedSetup = setup;
        AdvancedThreads = 0;

        auto& rootFEN = setup.StartFEN;
        if (rootFEN == InitialFEN && setup.SetupMoves.size() == 0) {
            rootFEN = rootPosition.GetFEN();
        }
        //  TODO: fix setup.StartFEN too?

        ScoredMove rms[MoveListSize] = {};
        i32 size = Generate<GenLegal>(rootPosition, &rms[0], 0);

        for (auto t : Threads) {
            auto td = t->worker.get();
            td->Reset();

            td->RootMoves.clear();
            td->RootMoves.shrink_to_fit();
            td->RootMoves.reserve(size);
            for (i32 j = 0; j < size; j++) {
                td->RootMoves.push_back(RootMove(rms[j].move));
            }

            if (setup.UCISearchMoves.size() != 0) {
                //td.RootMoves = td.RootMoves.Where(x => setup.UCISearchMoves.Contains(x.Move)).ToList();
            }

            td->RootPosition.LoadFromFEN(rootFEN);

            for (auto& move : setup.SetupMoves) {
                td->RootPosition.MakeMove(move);
            }
        }

        MainThreadBase()->WakeUp();
    }

    void SearchThreadPool::WaitForMain() const { MainThreadBase()->WaitForThreadFinished(); };
    SearchThread* SearchThreadPool::GetBestThread() const { return MainThread(); }

    void SearchThreadPool::SendBestMove() const {
        const auto td = GetBestThread();
        const auto bm = td->RootMoves[0].move;
        const auto bmStr = bm.SmithNotation(td->RootPosition.IsChess960);
        std::cout << "bestmove " << bmStr << std::endl;
    }

    void SearchThreadPool::AwakenHelperThreads() const {
        for (i32 i = 1; i < Threads.size(); i++)
            Threads[i]->WakeUp();
    }

    void SearchThreadPool::WaitForSearchFinished() const {
        for (i32 i = 1; i < Threads.size(); i++)
            Threads[i]->WaitForThreadFinished();
    }

    void SearchThreadPool::Clear() const {
        for (i32 i = 0; i < Threads.size(); i++)
            Threads[i]->worker.get()->History.Clear();

        MainThread()->Nodes = 0;
    }

    bool SearchThreadPool::HasConsensus() const {
        if (Horsie::Threads == 1)
            return false;

        for (const auto& v : PoolHistory) {
            //  RootDepth <= 2
            if (v.size() <= 2)
                return false;

            if (v.back().PV.size() < 2)
                return false;
        }

        const auto& mainVec = PoolHistory[0].back();
        const auto m0 = mainVec.PV[0];
        const auto m1 = mainVec.PV[1];
        i32 agree = 1;

        for (i32 i = 1; i < PoolHistory.size(); i++) {
            const auto& tVec = PoolHistory[i].back();

            if (tVec.PV[0] == m0 && tVec.PV[1] == m1)
                agree++;
        }

        return agree > (PoolHistory.size() / 2);
    }

	bool SearchThread::ShouldStop() const {
        return StopSearching.load(std::memory_order::relaxed);
	}

	void SearchThread::SetStop(bool flag) {
        StopSearching.store(flag, std::memory_order::relaxed);
	}

    void SearchThread::CheckLimits() {
        if (IsDatagen) {
            if (Nodes >= HardNodeLimit && RootDepth > 2) {
                SetStop();
            }
        }
        else {
            assert(IsMain());

            if ((Nodes & CheckupFrequency) == CheckupFrequency && HardTimeReached()) {
                AssocPool->StopAllThreads();
            }

            //  Obey node limit exactly when single-threaded, or check it wrt. CheckupFrequency otherwise
            if ((Horsie::Threads == 1 && Nodes >= HardNodeLimit)
                || (Nodes & CheckupFrequency) == CheckupFrequency && AssocPool->GetNodeCount() >= HardNodeLimit) {
                AssocPool->StopAllThreads();
            }
        }
    }

    void SearchThread::UpdatePoolHistory(RootMove& rm) {
        AssocPool->PoolHistory[ThreadIdx].push_back(rm);
    }

    void SearchThread::AdvanceWithConsensus() {
        SaveRootMoves();

        const auto& rm = AssocPool->PoolHistory[ThreadIdx].back();
        const auto& pv = rm.PV;

        RootPosition.MakeMove(pv[0]);
        RootPosition.MakeMove(pv[1]);

        ScoredMove rms[MoveListSize] = {};
        i32 size = Generate<GenLegal>(RootPosition, &rms[0], 0);

        RootMoves.clear();
        RootMoves.shrink_to_fit();
        RootMoves.reserve(size);
        for (i32 j = 0; j < size; j++) {
            RootMoves.push_back(RootMove(rms[j].move));
            if (pv.size() > 2 && RootMoves[j].move == pv[2]) {
                std::swap(RootMoves[0], RootMoves[j]);
            }
        }
    }

    void SearchThread::UndoAdvance() {
        const auto& setup = AssocPool->SharedSetup;

        const auto& rootFEN = setup.StartFEN;
        RootPosition.LoadFromFEN(rootFEN);

        for (auto& move : setup.SetupMoves) {
            RootPosition.MakeMove(move);
        }

        RestoreRootMoves();
    }

    void SearchThread::SaveRootMoves() {
        SavedRootMoves.assign(RootMoves.begin(), RootMoves.end());
    }
    
    void SearchThread::RestoreRootMoves() {
        RootMoves.assign(SavedRootMoves.begin(), SavedRootMoves.end());
    }

	Thread::Thread(i32 n) {
		worker = std::make_unique<SearchThread>();
		_SysThread = std::thread(&Thread::IdleLoop, this);
	}

    Thread::~Thread() {
        Quit = true;
        WakeUp();
        _SysThread.join();
    }

    void Thread::WakeUp() {
        _Mutex.lock();
        searching = true;
        _Mutex.unlock();
        _SearchCond.notify_one();
    }

    void Thread::WaitForThreadFinished() {
        std::unique_lock<std::mutex> lk(_Mutex);
        _SearchCond.wait(lk, [&] { return !searching; });
    }

    void Thread::IdleLoop() {
        while (true) {
            std::unique_lock<std::mutex> lk(_Mutex);
            searching = false;
            _SearchCond.notify_one();
            _SearchCond.wait(lk, [&] { return searching; });

            if (Quit)
                return;

            lk.unlock();

            if (worker->IsMain()) {
                worker->MainThreadSearch();
            }
            else {
                worker->Search(worker->AssocPool->SharedInfo);
            }
        }
    }
}
