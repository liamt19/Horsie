#pragma once

#include "defs.h"
#include "move.h"

namespace Horsie {

    struct MoveList {

        MoveList() = default;

        inline ScoredMove& operator[](u32 i) { return List[i]; }

        inline auto Size() const { return Length; }
        inline void Clear() { Length = 0; }
        inline void Resize(u32 newSize) { Length = newSize; }

        inline void Add(Move m) { List[Length++].move = m; }
        inline void Add(ScoredMove sm) { List[Length++] = sm; }

        inline void Swap(u32 a, u32 b) { std::swap(List[a], List[b]); }

    private:
        std::array<ScoredMove, MoveListSize> List;
        u32 Length{};
    };

    enum MoveGenType {
        PseudoLegal,
        GenNoisy,
        GenEvasions,
        GenNonEvasions,
        GenLegal
    };

    class Position;

    template<MoveGenType>
    void Generate(const Position& pos, MoveList& moveList);
    void GenerateQS(const Position& pos, MoveList& moveList);
}
