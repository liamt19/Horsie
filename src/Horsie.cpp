

#include <iostream>
#include "zobrist.h"
#include "precomputed.h"
#include "position.h"
#include "movegen.h"

using namespace Horsie;

void handler(int sig);
int main();

int main()
{
    std::cout << "Hi" << std::endl;

    Precomputed::init();
    std::cout << "Precomputed::init()" << std::endl;

    Zobrist::init();
    std::cout << "Zobrist::init() done" << std::endl;

    SearchThread thread = SearchThread();

    Position pos = Position(InitialFEN, &thread);
    std::cout << "Created pos" << std::endl;

    ScoredMove movelist[MoveListSize];
    ScoredMove* list = &movelist[0];
    int size = Generate<GenLegal>(pos, list, 0);
    std::cout << "GenAll returned " << size << std::endl;
    for (size_t i = 0; i < size; i++)
    {
        //std::cout << "Move[" << i << "]: " << list[i].Move.SmithNotation(pos.IsChess960) << std::endl;
    }
    
    //pos.Perft(3);

    for (int i = 1; i < 7; i++)
    {
        ulong u = pos.Perft(i);
	    std::cout << "Perft(" << i << "): " << u << std::endl;
    }
    


    std::cin.get();
    return 0;
}