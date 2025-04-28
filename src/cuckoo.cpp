
#include "cuckoo.h"

#include "defs.h"
#include "enums.h"
#include "move.h"
#include "precomputed.h"
#include "types.h"
#include "zobrist.h"

#include <array>
#include <cassert>
#include <utility>

using namespace Horsie;

namespace Horsie::Cuckoo {

    std::array<u64, 8192> keys{};
    std::array<Move, 8192> moves{};

    void Init() {
        i32 count = 0;
        for (Color pc : {Color::WHITE, Color::BLACK}) {
            for (i32 pt = Piece::HORSIE; pt <= Piece::KING; ++pt) {
                for (Square s1 = Square::A1; s1 <= Square::H8; ++s1) {
                    for (Square s2 = s1 + 1; s2 <= Square::H8; ++s2) {
                        if ((attacks_bb(pt, s1, 0) & SquareBB(s2)) != 0) {
                            Move m = Move(s1, s2);
                            u64 key = Zobrist::PSQHashes[pc][pt][s1] ^ Zobrist::PSQHashes[pc][pt][s2] ^ Zobrist::BlackHash;

                            i32 iter = Hash1(key);
                            while (true) {
                                std::swap(keys[iter], key);
                                std::swap(moves[iter], m);

                                if (m.IsNull())
                                    break;

                                iter = iter == Hash1(key) ? Hash2(key) : Hash1(key);
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
