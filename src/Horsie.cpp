
#include "cuckoo.h"
#include "nnue/nn.h"
#include "precomputed.h"
#include "uci.h"
#include "zobrist.h"

#include <iostream>

using namespace Horsie;

i32 main(i32 argc, char* argv[]) {

#if defined(VS_COMP)
#undef EVALFILE
#define EVALFILE "conv1.bin"
#endif

    NNUE::LoadNetwork(std::string(EVALFILE));
    Precomputed::Init();
    Zobrist::Init();
    Cuckoo::Init();

    std::cout << "Horsie " << EngVersion << std::endl << std::endl;

    UCI::UCIClient uci{};
    uci.InputLoop(argc, argv);

    return 0;
}
