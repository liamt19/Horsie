

#include "threadpool.h"
#include "movegen.h"

namespace Horsie {

	void SearchThreadPool::Resize(int newThreadCount) {

		if (Threads.size() > 0) {
			MainThreadBase()->WaitForThreadFinished();

			while (Threads.size() > 0)
				delete Threads.back(), Threads.pop_back();
		}

		//Threads.clear();
		//Threads.resize(newThreadCount);
		
		for (i32 i = 0; i < newThreadCount; i++) {
			auto td = new Thread(i);
			auto worker = td->worker.get();
			worker->ThreadIdx = i;
			worker->AssocPool = this;
			worker->TT = &TTable;
			Threads.push_back(td);
		}

		MainThreadBase()->WaitForThreadFinished();
	}

	void SearchThreadPool::StartSearch(Position& rootPosition, const SearchLimits& rootInfo) {
		ThreadSetup setup{};
		StartSearch(rootPosition, rootInfo, setup);
	}

	void SearchThreadPool::StartSearch(Position& rootPosition, const SearchLimits& rootInfo, ThreadSetup& setup) {
		MainThreadBase()->WaitForThreadFinished();
		MainThread()->StartTime = std::chrono::system_clock::now();

		StopThreads.store(false, std::memory_order::seq_cst);
		SharedInfo = rootInfo;          //  Initialize the shared SearchInformation
		//SharedInfo.SearchActive = true;

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

		//SharedInfo.TimeManager.StartTimer();
		MainThreadBase()->start_searching();
	}

	SearchThread* SearchThreadPool::GetBestThread() const {
		return MainThread();
	}

	void SearchThreadPool::StartThreads() const {
		for (i32 i = 1; i < Threads.size(); i++) {
			Threads[i]->start_searching();
		}
	}

	void SearchThreadPool::WaitForSearchFinished() const {
		for (i32 i = 1; i < Threads.size(); i++) {
			Threads[i]->WaitForThreadFinished();
		}
	}

	void SearchThreadPool::BlockCallerUntilFinished() {
		Blocker.arrive_and_wait();
	}

	void SearchThreadPool::Clear() const {
		for (i32 i = 0; i < Threads.size(); i++) {
			Threads[i]->worker.get()->History.Clear();
		}

		MainThread()->CheckupCount = 0;
	}



	Thread::Thread(i32 n) {
		idx = n;
		worker = std::make_unique<SearchThread>();
		stdThread = std::thread(&Thread::idle_loop, this);
	}


	// Destructor wakes up the thread in idle_loop() and waits
	// for its termination. Thread should be already waiting.
	Thread::~Thread() {

		assert(!searching);

		exit = true;
		start_searching();
		stdThread.join();
	}


	// Wakes up the thread that will start the search
	void Thread::start_searching() {
		mutex.lock();
		searching = true;
		mutex.unlock();   // Unlock before notifying saves a few CPU-cycles
		cv.notify_one();  // Wake up the thread in idle_loop()
	}


	// Blocks on the condition variable
	// until the thread has finished searching.
	void Thread::WaitForThreadFinished() {

		std::unique_lock<std::mutex> lk(mutex);
		cv.wait(lk, [&] { return !searching; });
	}


	// Thread gets parked here, blocked on the
	// condition variable, when it has no work to do.

	void Thread::idle_loop() {
		while (true) {
			std::unique_lock<std::mutex> lk(mutex);
			searching = false;
			cv.notify_one();  // Wake up anyone waiting for search finished
			cv.wait(lk, [&] { return searching; });

			if (exit)
				return;

			lk.unlock();

			//worker->start_searching();
			if (worker->IsMain()) {
				worker->MainThreadSearch();
			}
			else {
				worker->Search(worker->AssocPool->SharedInfo);
			}
		}
	}
}