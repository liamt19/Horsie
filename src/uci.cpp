
#include "uci.h"

#include "cuckoo.h"
#include "datagen/selfplay.h"
#include "movegen.h"
#include "nnue/nn.h"
#include "position.h"
#include "precomputed.h"
#include "search_bench.h"
#include "threadpool.h"
#include "tt.h"
#include "util/timer.h"
#include "zobrist.h"

#include <chrono>
#include <iostream>
#include <list>
#include <optional>
#include <thread>

#if defined(PERM_COUNT)
#include <fstream>
#include <iostream>
#include <vector>
#endif

using namespace Horsie;
using namespace Horsie::Search;
using namespace Horsie::NNUE;
using namespace Horsie::Cuckoo;

namespace Horsie::UCI {

    void UCIClient::InputLoop(i32 argc, char* argv[]) {
        std::string token, cmd;

        if (argc > 1) {
            std::string arg1 = std::string(argv[1]);
            if (arg1 == "bench") {
                i32 depth = 12;
                if (argc > 2) {
                    depth = std::stoi(std::string(argv[2]));
                }

                Horsie::DoBench(*SearchPool, depth, true);
                return;
            }
        }

        do {
            if (!getline(std::cin, cmd))
                cmd = "quit";

            std::istringstream is(cmd);

            token.clear();
            is >> std::skipws >> token;

            if (token == "quit")
                break;

            else if (token == "uci")
                HandleUCICommand();

            else if (token == "setoption")
                HandleSetOptionCommand(is);

            else if (token == "ucinewgame")
                HandleNewGameCommand();

            else if (token == "position")
                HandleSetPosition(is);

            else if (token == "isready")
                HandleIsReadyCommand();

            else if (token == "go")
                HandleGoCommand(is);

            else if (token == "stop")
                HandleStopCommand();


            else if (token == "d")
                HandleDisplayPosition();

            else if (token == "eval")
                HandleEvalCommand();

            else if (token == "wait")
                HandleWaitCommand();


            else if (token == "bench")
                HandleBenchCommand(is);

            else if (token == "benchperft" || token == "b")
                HandleBenchPerftCommand();

            else if (token == "perft")
                HandlePerftCommand(is);

            else if (token == "list")
                HandleListMovesCommand();

            else if (token == "move")
                HandleMoveCommand(is);


            else if (token == "threads")
                HandleThreadsCommand(is);

            else if (token == "hash")
                HandleHashCommand(is);

            else if (token == "multipv")
                HandleMultiPVCommand(is);


            else if (token == "activations")
                HandlePrintActivations();

            else if (token == "tune")
                HandleTuneCommand();

            else if (token == "datagen")
                HandleDatagenCommand(is);


            else if (std::ranges::count(token, '/') == 7) {
                //  Reset the stream so the beginning part of the fen isn't consumed before we call SetPosition
                std::istringstream is(cmd);
                HandleSetPosition(is, true);
            }

        } while (true);
    }


    SearchLimits UCIClient::ParseGoParameters(std::istringstream& is) {
        SearchLimits limits = SearchLimits();
        std::string token;

        limits.MaxDepth = MaxDepth;
        limits.MaxNodes = UINT64_MAX;
        limits.MaxSearchTime = INT32_MAX;
        limits.MoveTime = 0;
        limits.Increment = 0;

        while (is >> token)
            if ((pos.ToMove == WHITE && token == "wtime") || (pos.ToMove == BLACK && token == "btime"))
                is >> limits.PlayerTime;

            else if ((pos.ToMove == WHITE && token == "winc") || (pos.ToMove == BLACK && token == "binc"))
                is >> limits.Increment;

            else if (token == "depth")
                is >> limits.MaxDepth;

            else if (token == "nodes")
                is >> limits.MaxNodes;

            else if (token == "movetime")
                is >> limits.MoveTime;

            else if (token == "movestogo")
                is >> limits.MovesToGo;

        return limits;
    }


    void UCIClient::HandleUCICommand() {
        std::cout << "id name Horsie " << EngVersion << std::endl;
        std::cout << "id author Liam McGuire" << std::endl;
        const auto& opts = GetUCIOptions();
        for (auto& opt : opts) {
            std::cout << opt << std::endl;
        }
        std::cout << "uciok" << std::endl;
        inUCI = true;

    }

    void UCIClient::HandleSetOptionCommand(std::istringstream& is) {
        std::string rawName{}, value{};
        is >> rawName;
        is >> rawName;

        is >> value;
        is >> value;

        std::string name = rawName;
        std::transform(name.begin(), name.end(), name.begin(), [](auto c) { return std::tolower(c); });

        auto opt = FindUCIOption(name);

        if (!opt) return;

        i32 newVal = (value == "false") ? 0
            : (value == "true") ? 1
            : std::stoi(value);

        if (newVal < opt->MinValue || newVal > opt->MaxValue) return;

        opt->CurrentValue = newVal;

        if (name == "hash") {
            SearchPool->TTable.Initialize(Horsie::Hash.CurrentValue);
            std::cout << "info string set hash to " << Horsie::Hash.CurrentValue << std::endl;
        }
        else if (name == "threads") {
            SearchPool->Resize(Horsie::Threads.CurrentValue);
            std::cout << "info string set threads to " << Horsie::Threads.CurrentValue << std::endl;
        }
    }

    void UCIClient::HandleNewGameCommand() {
        pos.LoadFromFEN(InitialFEN);
        SearchPool->Clear();
        SearchPool->TTable.Clear();
    }

    void UCIClient::HandleSetPosition(std::istringstream& is, bool skipToken) {
        std::string token, fen;

        if (skipToken) {
            token = "fen";
        }
        else {
            is >> token;
        }

        if (token == "startpos") {
            fen = InitialFEN;
            is >> token;  // Consume the "moves" token, if any
        }
        else {
            if (token != "fen") {
                //  Handle either "position fen <...>" or "position <...>"
                fen += token + " ";
            }

            while (is >> token && token != "moves")
                fen += token + " ";
        }

        pos.IsChess960 = Horsie::UCI_Chess960;
        pos.LoadFromFEN(fen);

        setup.StartFEN = fen;
        setup.SetupMoves.clear();

        while (is >> token) {
            bool found = false;
            Move m = pos.TryFindMove(token, found);
            if (found) {
                pos.MakeMove(m);
                setup.SetupMoves.push_back(m);
            }
        }
    }

    void UCIClient::HandleIsReadyCommand() {
        std::cout << "readyok" << std::endl;
    }

    void UCIClient::HandleGoCommand(std::istringstream& is) {
        auto thread = SearchPool->MainThread();
        if (!inUCI) {
            thread->History.Clear();
            SearchPool->TTable.Clear();
        }

        SearchLimits limits = ParseGoParameters(is);
        thread->SoftTimeLimit = limits.SetTimeLimits();

        SearchPool->StartSearch(pos, limits);
    }

    void UCIClient::HandleStopCommand() {
        SearchPool->StopAllThreads();
    }


    void UCIClient::HandleDisplayPosition() {
        std::cout << pos << std::endl;
    }

    void UCIClient::HandleEvalCommand() {
        std::cout << "Evaluation: " << NNUE::GetEvaluation(pos) << std::endl << std::endl;
    }

    void UCIClient::HandleWaitCommand() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }


    void UCIClient::HandleBenchCommand(std::istringstream& is) {
        i32 depth = ReadMaybe<i32>(is).value_or(12);
        Horsie::DoBench(*SearchPool, depth);
    }

    void UCIClient::HandleBenchPerftCommand() {
        const auto startTime = Timepoint::Now();

        u64 total = 0;
        for (std::string entry : EtherealFENs_D5) {
            std::string fen = entry.substr(0, entry.find(";"));
            std::string nodesStr = entry.substr(entry.find(";") + 1);

            u64 nodes = std::stoull(nodesStr);
            pos.LoadFromFEN(fen);

            u64 ourNodes = pos.Perft(5);
            if (ourNodes != nodes) {
                std::cout << "[" << fen << "] FAILED!! Expected: " << nodes << " Got: " << ourNodes << std::endl;
            }
            else {
                std::cout << "[" << fen << "] Passed" << std::endl;
            }

            total += ourNodes;
        }

        const auto duration = Timepoint::TimeSince(startTime);
        const auto [durSeconds, durMillis] = Timepoint::UnpackSecondsMillis(duration);
        const auto nps = Timepoint::NPS(total, duration);

        std::cout << "\nTotal: " << total << " in " << durSeconds << "." << durMillis << "s (" << FormatWithCommas(nps) << " nps)" << std::endl << std::endl;

    }

    void UCIClient::HandlePerftCommand(std::istringstream& is) {
        i32 depth = ReadMaybe<i32>(is).value_or(5);

        std::cout << "\nSplitPerft(" << depth << "): " << std::endl;

        const auto startTime = Timepoint::Now();
        u64 nodes = pos.SplitPerft(depth);
        const auto duration = Timepoint::TimeSince(startTime);
        const auto [durSeconds, durMillis] = Timepoint::UnpackSecondsMillis(duration);
        const auto nps = Timepoint::NPS(nodes, duration);

        std::cout << "\nTotal: " << nodes << " in " << durSeconds << "." << durMillis << "s (" << FormatWithCommas(nps) << " nps)" << std::endl << std::endl;
    }

    void UCIClient::HandleListMovesCommand() {
        ScoredMove pseudos[MoveListSize] = {};
        i32 pseudoSize = Generate<GenNonEvasions>(pos, &pseudos[0], 0);

        std::cout << "Pseudo: ";
        for (size_t i = 0; i < pseudoSize; i++)
            std::cout << Move::ToString(pseudos[i].move) << " ";
        std::cout << std::endl;

        ScoredMove legals[MoveListSize] = {};
        i32 legalsSize = Generate<GenLegal>(pos, &legals[0], 0);

        std::cout << "Legal: ";
        for (size_t i = 0; i < legalsSize; i++) {
            std::cout << Move::ToString(legals[i].move) << " ";
        }
        std::cout << std::endl;
    }

    void UCIClient::HandleMoveCommand(std::istringstream& is) {
        std::string moveStr = ReadMaybe<std::string>(is).value_or("a1a1");
        
        bool found{};
        Move m = pos.TryFindMove(moveStr, found);
        if (found)
            pos.MakeMove<true>(m);
    }


    void UCIClient::HandleThreadsCommand(std::istringstream& is) {
        i32 cnt = ReadMaybe<i32>(is).value_or(Horsie::Threads.DefaultValue);

        if (Horsie::Threads.TrySet(cnt)) {
            SearchPool->Resize(Horsie::Threads);
            std::cout << "info string set threads to " << cnt << std::endl;
        }
    }

    void UCIClient::HandleHashCommand(std::istringstream& is) {
        i32 cnt = ReadMaybe<i32>(is).value_or(Horsie::Hash.DefaultValue);

        if (Horsie::Hash.TrySet(cnt)) {
            SearchPool->TTable.Initialize(cnt);
            std::cout << "info string set hash to " << cnt << std::endl;
        }
    }

    void UCIClient::HandleMultiPVCommand(std::istringstream& is) {
        i32 cnt = ReadMaybe<i32>(is).value_or(Horsie::MultiPV.DefaultValue);

        if (Horsie::MultiPV.TrySet(cnt)) {
            std::cout << "info string set multipv to " << cnt << std::endl;
        }
    }


    void UCIClient::HandlePrintActivations() {
#if defined(PERM_COUNT)
        std::ofstream file("perm.txt");
        for (size_t i = 0; i < NNZCounts.size(); ++i) {
            file << i << " " << NNZCounts[i] << "\n";
        }
        file.close();

        std::cout << ActivationCount << " / " << EvalCalls << " = " << ((double)ActivationCount / EvalCalls) << std::endl;

        std::vector<std::pair<size_t, uint64_t>> indexedCounts;

        // Create the index-value pairs
        for (size_t i = 0; i < NNZCounts.size(); ++i) {
            indexedCounts.emplace_back(i, NNZCounts[i]);
        }

        // Filter pairs where index is less than L1_SIZE / 2
        indexedCounts.erase(
            std::remove_if(indexedCounts.begin(), indexedCounts.end(), [&](const auto& pair) { return pair.first >= L1_SIZE / 2; }), indexedCounts.end());

        // Sort by value in descending order
        std::sort(indexedCounts.begin(), indexedCounts.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

        std::vector<size_t> indices;
        std::transform(indexedCounts.begin(), indexedCounts.end(), std::back_inserter(indices), [](const auto& pair) { return pair.first; });

        // Chunk the indices into groups of 16 and print
        const size_t chunkSize = 16;
        for (size_t i = 0; i < indices.size(); i += chunkSize) {
            std::ostringstream oss;
            for (size_t j = i; j < i + chunkSize && j < indices.size(); ++j) {
                if (j > i) oss << ", ";
                oss << indices[j];
            }
            std::cout << oss.str() << ",\n";
        }
#endif
    }

    void UCIClient::HandleTuneCommand() {
        auto& opts = GetUCIOptions();

        for (auto& opt : opts) {
            if (opt.HideTune)
                continue;

            auto step = std::max(0.01, (opt.MaxValue - opt.MinValue) / 10.0);
            auto lr = std::max(0.002, 0.002 * (0.50 / step));

            auto dispStep = (static_cast<i32>(step * 10) / 10.0);
            auto dispLr = (static_cast<i32>(lr * 10000) / 10000.0);
            std::cout << opt.Name << ", int, " << opt.DefaultValue << ", " << opt.MinValue << ", " << opt.MaxValue << ", " << dispStep << ", " << dispLr << std::endl;

        }
    }


    void UCIClient::HandleDatagenCommand(std::istringstream& is) {
        std::string token;
        std::vector<std::string> tokens{};
        while (is >> token) {
            tokens.push_back(token);
        }

        auto findOption = [&tokens](const std::string& opt) -> std::optional<u64> {
            auto it = std::find_if(tokens.begin(), tokens.end(), [opt](const std::string& str) { return !str.empty() && str.back() == opt.front(); });

            if (it != tokens.end()) {
                std::string numberPart = it->substr(0, it->size() - 1);
                return std::stoull(numberPart);
            }

            return {};
        };

        u64 nodes = findOption("nodes").value_or(Datagen::SoftNodeLimit);
        u64 depth = findOption("depth").value_or(Datagen::DepthLimit);
        u64 games = findOption("games").value_or(Datagen::DefaultGameCount);
        u64 threads = findOption("threads").value_or(Datagen::DefaultThreadCount);
        bool dfrc = tokens.size() > 0 && tokens.back() == "dfrc";

        std::cout << std::format("Threads:      {}\n", threads);
        std::cout << std::format("Games/thread: {}\n", games);
        std::cout << std::format("Total games:  {}\n", games * threads);
        std::cout << std::format("Node limit:   {}\n", nodes);
        std::cout << std::format("Depth limit:  {}\n", depth);
        std::cout << std::format("Variant:      {}\n", (dfrc ? "DFRC" : "Standard"));
        std::cout << std::format("Hit enter to begin...\n");
        std::cin.get();

        Datagen::MonitorDG(nodes, depth, games, threads, dfrc);
    }

}