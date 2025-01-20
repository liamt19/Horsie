
#include "cuckoo.h"

#include "zobrist.h"
#include "precomputed.h"

#include <algorithm>

using namespace Horsie;

namespace Horsie::Cuckoo {

    std::array<u64, 8192> keys{};
    std::array<Move, 8192> moves{};

    void init() {
        i32 count = 0;
        for (i32 pc = Color::WHITE; pc <= Color::BLACK; pc++) {
            for (i32 pt = Piece::HORSIE; pt <= Piece::KING; pt++) {
                for (i32 s1 = 0; s1 < 64; s1++) {
                    for (i32 s2 = s1 + 1; s2 < 64; s2++) {
                        if ((attacks_bb(pt, s1, 0) & SquareBB(s2)) != 0) {
                            Move m = Move(s1, s2);
                            u64 key = Zobrist::ColorPieceSquareHashes[pc][pt][s1] ^ Zobrist::ColorPieceSquareHashes[pc][pt][s2] ^ Zobrist::BlackHash;

                            i32 iter = hash1(key);
                            while (true) {
                                std::swap(keys[iter], key);
                                std::swap(moves[iter], m);

                                if (m.IsNull())
                                    break;

                                iter = iter == hash1(key) ? hash2(key) : hash1(key);
                            }
                            ++count;
                        }
                    }
                }
            }
        }

        assert(count == 3668);
    }

}