

#include "threadpool.h"
#include "movegen.h"

namespace Horsie {

	void SearchThreadPool::Resize(int newThreadCount) {

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

		WaitForMain();
	}

	void SearchThreadPool::StartSearch(Position& rootPosition, const SearchLimits& rootInfo) {
		ThreadSetup setup{};
		StartSearch(rootPosition, rootInfo, setup);
	}

	void SearchThreadPool::StartSearch(Position& rootPosition, const SearchLimits& rootInfo, ThreadSetup& setup) {
		WaitForMain();
		MainThread()->StartTime = std::chrono::system_clock::now();

		StopThreads = false;
		SharedInfo = rootInfo;

		auto& rootFEN = setup.StartFEN;
		if (rootFEN == InitialFEN && setup.SetupMoves.size() == 0) {
			rootFEN = rootPosition.GetFEN();
		}

		ScoredMove rms[MoveListSize] = {};
		i32 size = Generate<GenLegal>(rootPosition, &rms[0], 0);

		for (auto t : Threads) {
			auto td = t->worker.get();
			td->Reset();

			td->RootMoves.clear();
			td->RootMoves.shrink_to_fit();
			td->RootMoves.reserve(size);
			for (int j = 0; j < size; j++) {
				td->RootMoves.push_back(RootMove(rms[j].move));
			}

			if (setup.UCISearchMoves.size() != 0) {
				//td.RootMoves = td.RootMoves.Where(x => setup.UCISearchMoves.Contains(x.Move)).ToList();
			}

			td->RootPosition.LoadFromFEN(rootFEN);

			for(auto& move : setup.SetupMoves) {
				td->RootPosition.MakeMove(move);
			}
		}

		MainThreadBase()->start_searching();
	}

	void SearchThreadPool::WaitForMain() const { MainThreadBase()->WaitForThreadFinished(); };
	SearchThread* SearchThreadPool::GetBestThread() const { return MainThread(); }

	void SearchThreadPool::SendBestMove() const {
		const auto td = GetBestThread();
		const auto bm = td->RootMoves[0].move;
		const auto bmStr = bm.SmithNotation(td->RootPosition.IsChess960);
		std::cout << "bestmove " << bmStr << std::endl;
	}

	void SearchThreadPool::StartThreads() const {
		for (i32 i = 1; i < Threads.size(); i++)
			Threads[i]->start_searching();
	}

	void SearchThreadPool::WaitForSearchFinished() const {
		for (i32 i = 1; i < Threads.size(); i++)
			Threads[i]->WaitForThreadFinished();
	}

	void SearchThreadPool::Clear() const {
		for (i32 i = 0; i < Threads.size(); i++)
			Threads[i]->worker.get()->History.Clear();

		MainThread()->CheckupCount = 0;
	}


	Thread::Thread(i32 n) {
		worker = std::make_unique<SearchThread>();
		stdThread = std::thread(&Thread::idle_loop, this);
	}

	Thread::~Thread() {
		assert(!searching);
		exit = true;
		start_searching();
		stdThread.join();
	}


	void Thread::start_searching() {
		mutex.lock();
		searching = true;
		mutex.unlock();
		cv.notify_one();
	}

	void Thread::WaitForThreadFinished() {
		std::unique_lock<std::mutex> lk(mutex);
		cv.wait(lk, [&] { return !searching; });
	}

	void Thread::idle_loop() {
		while (true) {
			std::unique_lock<std::mutex> lk(mutex);
			searching = false;
			cv.notify_one();
			cv.wait(lk, [&] { return searching; });

			if (exit)
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