
#include <iostream>
#include "zobrist.h"
#include "precomputed.h"
#include "position.h"
#include "movegen.h"

using namespace Horsie;

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
    int size = Generate<GenLegal>(pos, list);
    std::cout << "GenAll returned " << size << std::endl;


    std::cin.get();
    return 0;
}
