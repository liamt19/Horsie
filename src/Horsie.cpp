
#include <chrono>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <list>
#include <thread>
#include <barrier>

#include "threadpool.h"
#include "zobrist.h"
#include "precomputed.h"
#include "cuckoo.h"
#include "position.h"
#include "movegen.h"
#include "tt.h"
#include "nnue/nn.h"
#include "search_bench.h"



using namespace Horsie;
using namespace Horsie::Search;
using namespace Horsie::NNUE;
using namespace Horsie::Cuckoo;

using std::cout;
using std::endl;

i32 main(i32 argc, char* argv[]);
void HandleSetPosition(Position& pos, std::istringstream& is, bool skipToken = false);
void HandleDisplayPosition(Position& pos);
void HandlePerftCommand(Position& pos, std::istringstream& is);
void HandleThreadsCommand(std::istringstream& is);
void HandleBenchCommand(std::istringstream& is);
void HandleSetOptionCommand(std::istringstream& is);
void HandleBenchPerftCommand(Position& pos);
void HandleMoveCommand(Position& pos, std::istringstream& is);
void HandleListMovesCommand(Position& pos);
SearchLimits ParseGoParameters(Position& pos, std::istringstream& is);
void HandleGoCommand(Position& pos, std::istringstream& is);
void HandleStopCommand();
void HandleEvalCommand(Position& pos);
void HandleUCICommand();
void HandleNewGameCommand(Position& pos);
void ScuffedChatGPTPrintActivations();


bool inUCI = false;
std::unique_ptr<SearchThreadPool> SearchPool;
std::barrier is_sync_barrier(2);
ThreadSetup setup;

#if defined(_MSC_VER) && !defined(EVALFILE)

#define EVALFILE "meow.bin"

#endif

i32 main(i32 argc, char* argv[])
{
#ifdef EVALFILE
    auto net = std::string(EVALFILE);
#else
    auto net = std::string("net.bin");
#endif

    NNUE::LoadNetwork(net);
    Precomputed::init();
    Zobrist::init();
    Cuckoo::init();

    SearchPool = std::make_unique<SearchThreadPool>(Horsie::Threads);

    Position pos = Position(InitialFEN);

    std::cout << "Horsie 0.0" << std::endl << std::endl;

    if (argc > 1) {
        std::string arg1 = std::string(argv[1]);
        if (arg1 == "bench") {
            i32 depth = 12;
            if (argc > 2) {
                depth = std::stoi(std::string(argv[2]));
            }

            Horsie::DoBench(*SearchPool, depth, true);
            return 0;
        }
    }

    std::string token, cmd;

    do
    {
        if (!getline(std::cin, cmd))  // Wait for an input or an end-of-file (EOF) indication
            cmd = "quit";

        std::istringstream is(cmd);

        token.clear();  // Avoid a stale if getline() returns nothing or a blank line
        is >> std::skipws >> token;

        if (token == "quit")
            break;

        if (token == "position")
            HandleSetPosition(pos, is);

        else if (token == "d")
            HandleDisplayPosition(pos);

        else if (token == "perft")
            HandlePerftCommand(pos, is);

        else if (token == "benchperft" || token == "b")
            HandleBenchPerftCommand(pos);

        else if (token == "bench")
            HandleBenchCommand(is);

        else if (token == "list")
            HandleListMovesCommand(pos);

        else if (token == "move")
            HandleMoveCommand(pos, is);

        else if (token == "threads")
            HandleThreadsCommand(is);

        else if (token == "go") {
            HandleGoCommand(pos, is);
        }

        else if (token == "wait") {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        else if (token == "eval")
            HandleEvalCommand(pos);

        else if (token == "uci")
            HandleUCICommand();

        else if (token == "isready")
            std::cout << "readyok" << std::endl;

        else if (token == "stop")
            HandleStopCommand();

        else if (token == "ucinewgame")
            HandleNewGameCommand(pos);

        else if (token == "setoption")
            HandleSetOptionCommand(is);

        else if (token == "activations")
            ScuffedChatGPTPrintActivations();
        
        else if (std::ranges::count(token, '/') == 7) {
            //  Reset the stream so the beginning part of the fen isn't consumed before we call SetPosition
            std::istringstream is(cmd);
            HandleSetPosition(pos, is, true);
        }

    } while (true);

    return 0;
}



void HandleSetPosition(Position& pos, std::istringstream& is, bool skipToken) {
    std::string token, fen;

    if (skipToken) {
        token = "fen";
    }
    else {
        is >> token;
    }

    if (token == "startpos")
    {
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

    while (is >> token)
    {
        bool found = false;
        Move m = pos.TryFindMove(token, found);
        if (found) {
            pos.MakeMove(m);
            setup.SetupMoves.push_back(m);
        }
        else {
            std::cout << "Move " << token << " not found!" << std::endl;
        }
    }

}




void HandleSetOptionCommand(std::istringstream& is) {
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
               : (value == "true")  ? 1
               :                      std::stoi(value);

    if (newVal < opt->MinValue || newVal > opt->MaxValue) return;

    opt->CurrentValue = newVal;

    if (name == "hash") {
        SearchPool->TTable.Initialize(Horsie::Hash);
        std::cout << "info string set hash to " << Horsie::Hash << std::endl;
    }
    else if (name == "threads") {
        SearchPool->Resize(Horsie::Threads);
        std::cout << "info string set threads to " << Horsie::Threads << std::endl;
    }
}


void HandleMoveCommand(Position& pos, std::istringstream& is) {
    
    std::string moveStr;
    if (is && is.peek() != EOF)
        is >> moveStr;
    bool found;
    Move m = pos.TryFindMove(moveStr, found);
    if (found) {
        pos.MakeMove<true>(m);
    }
    else {
        cout << "Failed to find '" << moveStr << "'" << endl;
    }
}




SearchLimits ParseGoParameters(Position& pos, std::istringstream& is) {
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


void HandleGoCommand(Position& pos, std::istringstream& is) {

    auto thread = SearchPool->MainThread();
    if (!inUCI) {
        thread->History.Clear();
        SearchPool->TTable.Clear();
    }

    SearchLimits limits = ParseGoParameters(pos, is);

    thread->SoftTimeLimit = limits.SetTimeLimits();

    SearchPool->StartSearch(pos, limits);
}



void HandleEvalCommand(Position& pos) {
    cout << "Evaluation: " << NNUE::GetEvaluation(pos) << endl << endl;

    ScoredMove legals[MoveListSize] = {};
    i32 legalsSize = Generate<GenLegal>(pos, &legals[0], 0);
    std::list<ScoredMove> moves;

    for (size_t i = 0; i < legalsSize; i++)
    {
        Move m = legals[i].move;
        pos.MakeMove(m);
        i32 eval = NNUE::GetEvaluation(pos);
        pos.UnmakeMove(m);

        moves.push_back({ m, (eval * -1) });
    }

    moves.sort([](const ScoredMove& l, const ScoredMove& r) { return l.Score > r.Score; });

    for (ScoredMove m : moves) {
        cout << Move::ToString(m.move) << ": " << m.Score << endl;
    }
}




void HandleStopCommand() {
    SearchPool->SetStop();
}



void HandleUCICommand() {
    
    std::cout << "id name Horsie" << std::endl;
    const auto& opts = GetUCIOptions();
    for (auto& opt : opts) {
        std::cout << opt << std::endl;
    }
    std::cout << "uciok" << std::endl;
    inUCI = true;

}



void HandleNewGameCommand(Position& pos) {
    pos.LoadFromFEN(InitialFEN);
    SearchPool->Clear();
    SearchPool->TTable.Clear();
}


void HandleDisplayPosition(Position& pos) {
    cout << pos << endl;
}


void HandlePerftCommand(Position& pos, std::istringstream& is) {
    i32 depth = 5;

    if (is && is.peek() != EOF)
        is >> depth;

    cout << "\nSplitPerft(" << depth << "): " << endl;
    auto timeStart = std::chrono::high_resolution_clock::now();
    u64 nodes = pos.SplitPerft(depth);
    auto timeEnd = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
    auto dur = duration.count();
    auto durSeconds = duration.count() / 1000;
    auto durMillis = duration.count() % 1000;

    auto nps = (nodes / ((double)dur / 1000));
    cout << "\nTotal: " << nodes << " in " << durSeconds << "." << durMillis << "s (" << FormatWithCommas((u64)nps) << " nps)" << endl << endl;
}


void HandleBenchPerftCommand(Position& pos) {
    auto timeStart = std::chrono::high_resolution_clock::now();

    u64 total = 0;
    for (std::string entry : EtherealFENs_D5)
    {
        std::string fen = entry.substr(0, entry.find(";"));
        std::string nodesStr = entry.substr(entry.find(";") + 1);

        u64 nodes = std::stoull(nodesStr);
        pos.LoadFromFEN(fen);

        u64 ourNodes = pos.Perft(5);
        if (ourNodes != nodes) {
            cout << "[" << fen << "] FAILED!! Expected: " << nodes << " Got: " << ourNodes << endl;
        }
        else {
            cout << "[" << fen << "] Passed" << endl;
        }

        total += ourNodes;
    }

    auto timeEnd = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
    auto dur = duration.count();
    auto durSeconds = duration.count() / 1000;
    auto durMillis = duration.count() % 1000;

    auto nps = (total / ((double)dur / 1000));
    cout << "\nTotal: " << total << " in " << durSeconds << "." << durMillis << "s (" << FormatWithCommas((u64)nps) << " nps)" << endl << endl;

}


void HandleBenchCommand(std::istringstream& is) {

    i32 depth = 12;
    if (is && is.peek() != EOF)
        is >> depth;

    Horsie::DoBench(*SearchPool, depth);
}


void HandleListMovesCommand(Position& pos) {
    ScoredMove pseudos[MoveListSize] = {};
    i32 pseudoSize = Generate<GenNonEvasions>(pos, &pseudos[0], 0);

    cout << "Pseudo: ";
    for (size_t i = 0; i < pseudoSize; i++)
        cout << Move::ToString(pseudos[i].move) << " ";
    cout << endl;

    ScoredMove legals[MoveListSize] = {};
    i32 legalsSize = Generate<GenLegal>(pos, &legals[0], 0);

    cout << "Legal: ";
    for (size_t i = 0; i < legalsSize; i++)
    {
        cout << Move::ToString(legals[i].move) << " ";
    }
    cout << endl;
}


void HandleThreadsCommand(std::istringstream& is) {

    i32 cnt = 1;
    if (is && is.peek() != EOF)
        is >> cnt;

    if (cnt >= Horsie::Threads.MinValue && cnt <= Horsie::Threads.MaxValue) {
        Horsie::Threads = cnt;
        SearchPool->Resize(Horsie::Threads);
        std::cout << "info string set threads to " << Horsie::Threads << std::endl;
    }
}


#if defined(PERM_COUNT)
#include <fstream>
#include <iostream>
#include <vector>
#endif

void ScuffedChatGPTPrintActivations() {
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