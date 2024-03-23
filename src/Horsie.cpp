
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
    std::cout << pos << std::endl;
}



void HandlePerftCommand(Position& pos, std::istringstream& is) {
	int depth = 5;

    if (is && is.peek() != EOF)
	    is >> depth;

    std::cout << "\nSplitPerft(" << depth << "): " << std::endl;
    auto timeStart = std::chrono::high_resolution_clock::now();
    ulong nodes = pos.SplitPerft(depth);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - timeStart);
	 std::cout << "\nTotal: " << nodes << " in " << duration << std::endl << std::endl;
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
            std::cout << "[" << fen << "] FAILED!! Expected: " << nodes << " Got: " << ourNodes << std::endl;
        }
        else {
            std::cout << "[" << fen << "] Passed, Expected: " << nodes << " Got: " << ourNodes << std::endl;
        }
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - timeStart);
    std::cout << "Done in " << duration << std::endl;
}



void HandleListMovesCommand(Position& pos) {
    ScoredMove pseudos[MoveListSize] = {};
    int pseudoSize = Generate<GenNonEvasions>(pos, &pseudos[0], 0);

    std::cout << "Pseudo: ";
    for (size_t i = 0; i < pseudoSize; i++)
        std::cout << Move::ToString(pseudos[i].Move) << " ";
    std::cout << std::endl;

    ScoredMove legals[MoveListSize] = {};
    int legalsSize = Generate<GenLegal>(pos, &legals[0], 0);

    std::cout << "Legal: ";
    for (size_t i = 0; i < legalsSize; i++)
    {
        std::cout << Move::ToString(legals[i].Move) << " ";
    }
    std::cout << std::endl;
}