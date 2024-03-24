
#include <chrono>
#include <sstream>
#include <iostream>

#include "zobrist.h"
#include "precomputed.h"
#include "position.h"
#include "movegen.h"
#include "tt.h"

using namespace Horsie;
using namespace Horsie::Search;
using std::cout;
using std::endl;

int main();
void HandleSetPosition(Position& pos, std::istringstream& is);
void HandleDisplayPosition(Position& pos);
void HandlePerftCommand(Position& pos, std::istringstream& is);
void HandleBenchCommand(Position& pos);
void HandleListMovesCommand(Position& pos);
void HandleGoCommand(Position& pos, std::istringstream& is);


int main()
{
    Precomputed::init();
    Zobrist::init();
    TT.Initialize(16);

    Position pos = Position(InitialFEN);

    std::string token, cmd;

    do
    {
        if (!getline(std::cin, cmd))  // Wait for an input or an end-of-file (EOF) indication
            cmd = "quit";

        std::istringstream is(cmd);

        token.clear();  // Avoid a stale if getline() returns nothing or a blank line
        is >> std::skipws >> token;

        if (token == "quit" || token == "stop")
            break;

        if (token == "position")
            HandleSetPosition(pos, is);
        else if (token == "d")
            HandleDisplayPosition(pos);
        else if (token == "perft")
            HandlePerftCommand(pos, is);
        else if (token == "bench")
            HandleBenchCommand(pos);
        else if (token == "list")
            HandleListMovesCommand(pos);
        else if (token == "go")
            HandleGoCommand(pos, is);


    } while (true);

    std::cin.get();
    return 0;
}



void HandleSetPosition(Position& pos, std::istringstream& is) {
    std::string token, fen;

    is >> token;

    if (token == "startpos")
    {
        fen = InitialFEN;
        is >> token;  // Consume the "moves" token, if any
    }
    else {
        fen += token + " ";
        while (is >> token && token != "moves")
            fen += token + " ";
    }

    pos.LoadFromFEN(fen);
}



void HandleDisplayPosition(Position& pos) {
    cout << pos << endl;
}



void HandlePerftCommand(Position& pos, std::istringstream& is) {
    int depth = 5;

    if (is && is.peek() != EOF)
        is >> depth;

    cout << "\nSplitPerft(" << depth << "): " << endl;
    auto timeStart = std::chrono::high_resolution_clock::now();
    ulong nodes = pos.SplitPerft(depth);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - timeStart);
     cout << "\nTotal: " << nodes << " in " << duration << endl << endl;
}



void HandleBenchCommand(Position& pos) {
    auto timeStart = std::chrono::high_resolution_clock::now();

    for (std::string entry : EtherealFENs_D5)
    {
        std::string fen = entry.substr(0, entry.find(";"));
        std::string nodesStr = entry.substr(entry.find(";") + 1);

        ulong nodes = std::stoull(nodesStr);
        pos.LoadFromFEN(fen);

        ulong ourNodes = pos.Perft(5);
        if (ourNodes != nodes) {
            cout << "[" << fen << "] FAILED!! Expected: " << nodes << " Got: " << ourNodes << endl;
        }
        else {
            cout << "[" << fen << "] Passed, Expected: " << nodes << " Got: " << ourNodes << endl;
        }
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - timeStart);
    cout << "Done in " << duration << endl;
}



void HandleListMovesCommand(Position& pos) {
    ScoredMove pseudos[MoveListSize] = {};
    int pseudoSize = Generate<GenNonEvasions>(pos, &pseudos[0], 0);

    cout << "Pseudo: ";
    for (size_t i = 0; i < pseudoSize; i++)
        cout << Move::ToString(pseudos[i].Move) << " ";
    cout << endl;

    ScoredMove legals[MoveListSize] = {};
    int legalsSize = Generate<GenLegal>(pos, &legals[0], 0);

    cout << "Legal: ";
    for (size_t i = 0; i < legalsSize; i++)
    {
        cout << Move::ToString(legals[i].Move) << " ";
    }
    cout << endl;
}


void HandleGoCommand(Position& pos, std::istringstream& is) {
    SearchThread thread = SearchThread();
    thread.Reset();
    cout << "Calling Search" << endl;
    thread.Search(pos);

}