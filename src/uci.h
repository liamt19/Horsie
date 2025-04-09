#pragma once

#include "cuckoo.h"
#include "defs.h"
#include "movegen.h"
#include "nnue/nn.h"
#include "position.h"
#include "precomputed.h"
#include "search.h"
#include "search_bench.h"
#include "threadpool.h"
#include "tt.h"
#include "zobrist.h"

#include <chrono>
#include <iostream>
#include <list>
#include <thread>


namespace Horsie::UCI {

    class UCIClient {
    public:
        UCIClient() = default;
        void InputLoop(i32 argc, char* argv[]);

    private:
        Position pos{ InitialFEN };
        std::unique_ptr<SearchThreadPool> SearchPool{ std::make_unique<SearchThreadPool>(1) };
        ThreadSetup setup{};
        bool inUCI{};

        SearchLimits ParseGoParameters(std::istringstream& is);

        void HandleUCICommand();
        void HandleSetOptionCommand(std::istringstream& is);
        void HandleNewGameCommand();
        void HandleSetPosition(std::istringstream& is, bool skipToken = false);
        void HandleIsReadyCommand();
        void HandleGoCommand(std::istringstream& is);
        void HandleStopCommand();

        void HandleDisplayPosition();
        void HandleEvalCommand();
        void HandleWaitCommand();
        
        void HandleBenchCommand(std::istringstream& is);
        void HandleBenchPerftCommand();
        void HandlePerftCommand(std::istringstream& is);
        void HandleListMovesCommand();
        void HandleMoveCommand(std::istringstream& is);

        void HandleThreadsCommand(std::istringstream& is);
        void HandleHashCommand(std::istringstream& is);
        void HandleMultiPVCommand(std::istringstream& is);

        void HandlePrintActivations();
        void HandleTuneCommand();

        void HandleDatagenCommand(std::istringstream& is);
    };

}