
#include "cuckoo.h"
#include "precomputed.h"
#include "uci.h"
#include "zobrist.h"

#include <iostream>

using namespace Horsie;

i32 main(i32 argc, char* argv[]) {

    Precomputed::Init();
    Zobrist::Init();
    Cuckoo::Init();

    std::cout << "Horsie " << EngVersion << std::endl << std::endl;

    UCI::UCIClient uci{};
    uci.InputLoop(argc, argv);

    return 0;
}
