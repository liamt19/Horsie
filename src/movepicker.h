#pragma once

#include "defs.h"
#include "history.h"
#include "move.h"
#include "movegen.h"
#include "position.h"
#include "search.h"
#include "util.h"

#include <cassert>


namespace Horsie {

    namespace Search {
        enum class MovepickerStage : i32 {
            TT = 0,
            GenNoisies,
            GoodNoisies,
            Killer,
            GenQuiets,
            PlayQuiets,
            StartBadNoisies,
            BadNoisies,
            End,
        };


        inline MovepickerStage& operator++(MovepickerStage& v) {
            v = static_cast<MovepickerStage>(static_cast<i32>(v) + 1);
            return v;
        }


        struct ScoredMoveList {
            ScoredMove List[MoveListSize];
            i32 GeneratedSize{};

            constexpr auto Size() const { return GeneratedSize; }
            inline ScoredMove& operator[](i32 i) { return List[i]; }

            void Append(ScoredMove m) { List[GeneratedSize++] = m; }
            
            void RemoveAt(int i) {
                assert(i < GeneratedSize);
                GeneratedSize--;
                std::swap(List[i], List[GeneratedSize]);
            }

            constexpr auto& First() const { return List[0]; }
            constexpr auto& Last() const { return List[GeneratedSize]; }

            template <MoveGenType GenType>
            void GenerateInto(Position& pos);
        };



        class Movepicker {
        public:
            Movepicker(Position& pos, HistoryTable& history, Search::SearchStackEntry* ss, Move ttMove);

            Move Next();
            void ScoreList(ScoredMoveList& list);
            void StartSkippingQuiets();

        private:
            Position& Pos;
            HistoryTable& History;
            Search::SearchStackEntry* ss;
            Move TTMove{};
            Move KillerMove{};

            ScoredMoveList NoisyMoves;
            ScoredMoveList BadNoisyMoves;
            ScoredMoveList QuietMoves;

            i32 MoveIndex{};
            MovepickerStage Stage{};
            bool SkipQuiets{};
        };
    }

}