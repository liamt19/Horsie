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
            NMEnd = 8,

            QSearchTT = 10,
            QSearchGenNoisies,
            QSearchPlayNoisies,
            QSearchGenEvasions,
            QSearchEvasions,
            QSEnd = 15,
        };

        enum class MovepickerType : i32 {
            Negamax,
            QSearch,
            Probcut,
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

            std::string ToString() const {
                std::ostringstream oss;
                for (size_t i = 0; i < GeneratedSize; i++) {
                    oss << List[i].move << ": " << List[i].score;
                    if (i != GeneratedSize - 1)
                        oss << ", ";
                }
                return oss.str();
            }
        };



        class Movepicker {
        public:
            Movepicker(Position& pos, HistoryTable& history, MovepickerType genType, Search::SearchStackEntry* ss, Move ttMove);
            Move Next();
            void StartSkippingQuiets() { SkipQuiets = true; }
            bool FinishedGoodNoisies() const;

            static Movepicker Negamax(Position& pos, HistoryTable& history, Search::SearchStackEntry* ss, Move ttMove) {
                return Movepicker(pos, history, MovepickerType::Negamax, ss, ttMove);
            }

            static Movepicker QSearch(Position& pos, HistoryTable& history, Search::SearchStackEntry* ss, Move ttMove) {
                return Movepicker(pos, history, MovepickerType::QSearch, ss, ttMove);
            }


        private:

            ScoredMove OrderNext(ScoredMoveList& list);
            void ScoreList(ScoredMoveList& list);

            Position& Pos;
            HistoryTable& History;
            MovepickerType GenerationType;
            Search::SearchStackEntry* ss;
            Move TTMove{};
            Move KillerMove{};

            bool Evasions{};

            ScoredMoveList NoisyMoves;
            ScoredMoveList BadNoisyMoves;
            ScoredMoveList QuietMoves;

            i32 MoveIndex{};
            MovepickerStage Stage{};
            bool SkipQuiets{};
        };
    }

}