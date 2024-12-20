#pragma once


#ifndef SEARCH_H
#define SEARCH_H

#include <array>
#include <vector>
#include <chrono>
#include <algorithm>
#include <functional>

#include "move.h"
#include "history.h"
#include "util/alloc.h"

#include "search_options.h"

namespace Horsie {

    enum SearchNodeType {
        PVNode,
        NonPVNode,
        RootNode
    };

    class TranspositionTable;
    class Position;

    namespace Search {

        struct SearchLimits {
        public:
            i32 MaxDepth = Horsie::MaxDepth;
            u64 MaxNodes = UINT64_MAX;
            u64 SoftNodeLimit = UINT64_MAX;
            i32 MaxSearchTime = INT32_MAX;
            i32 Increment = 0;
            i32 MovesToGo = 20;

            i32 MoveTime = 0;
            const bool HasMoveTime() const { return MoveTime != 0; }

            i32 PlayerTime = 0;
            const bool HasPlayerTime() const { return PlayerTime != 0; }

            const bool IsInfinite() const { return MaxDepth == Horsie::MaxDepth && MaxSearchTime == INT32_MAX; }

            const i32 SetTimeLimits() {
                i32 softLimit = 0;

                if (HasMoveTime()) {
                    MaxSearchTime = MoveTime;
                }
                else if (HasPlayerTime()) {
                    MaxSearchTime = Increment + (PlayerTime / std::min(MovesToGo, 2));
                    MaxSearchTime = std::min(MaxSearchTime, PlayerTime);
                    softLimit = static_cast<i32>(0.65 * ((static_cast<double>(PlayerTime) / MovesToGo) + (Increment * 3 / 4.0)));
                }
                else {
                    MaxSearchTime = INT32_MAX;
                }

                return softLimit;
            }

            void PrintLimits() const {
				std::cout << "MaxDepth:      " << MaxDepth << std::endl;
				std::cout << "MaxNodes:      " << MaxNodes << std::endl;
				std::cout << "MaxSearchTime: " << MaxSearchTime << std::endl;
				std::cout << "Increment:     " << Increment << std::endl;
				std::cout << "MovesToGo:     " << MovesToGo << std::endl;
				std::cout << "MoveTime:      " << MoveTime << std::endl;
				std::cout << "PlayerTime:    " << PlayerTime << std::endl;
			}
        };

        struct SearchStackEntry {
        public:
            Move* PV;
            PieceToHistory* ContinuationHistory;
            i16 DoubleExtensions;
            i16 Ply;
            i16 StaticEval;
            Move KillerMove;
            Move CurrentMove;
            Move Skip;
            bool InCheck;
            bool TTPV;
            bool TTHit;
            i32 PVLength;


            void Clear() {
                CurrentMove = Skip = KillerMove = Move::Null();
                ContinuationHistory = nullptr;

                Ply = 0;
                DoubleExtensions = 0;
                StaticEval = ScoreNone;

                PVLength = 0;
                if (PV != nullptr)
                {
                    AlignedFree(PV);
                    PV = nullptr;
                }

                InCheck = false;
                TTPV = false;
                TTHit = false;
            }
        };

        struct RootMove {

            explicit RootMove(Move m) : PV(1, m) {
                move = m;
                Score = PreviousScore = AverageScore = -ScoreInfinite;
                Depth = 0;
            }
            bool operator==(const Move& m) const { return PV[0] == m; }
            bool operator<(const RootMove& m) const { return m.Score != Score ? m.Score < Score : m.PreviousScore < PreviousScore; }
            
            Move move;
            i32 Score;
            i32 PreviousScore;
            i32 AverageScore;
            i32 Depth;

            std::vector<Horsie::Move> PV;
        };

        struct ThreadSetup {
            std::string StartFEN;
            std::vector<Move> SetupMoves;
            std::vector<Move> UCISearchMoves;

            ThreadSetup(std::vector<Move> setupMoves) : ThreadSetup(InitialFEN, setupMoves, {}) {}
            ThreadSetup(std::string_view fen = InitialFEN) : ThreadSetup(fen, {}, {}) {}

            ThreadSetup(std::string_view fen, std::vector<Move> setupMoves, std::vector<Move> uciSearchMoves) {
                StartFEN = fen;
                SetupMoves = setupMoves;
                UCISearchMoves = uciSearchMoves;
            }
        };


        static void StableSort(std::vector<RootMove>& items, i32 offset = 0, i32 end = -1) {
            
            std::stable_sort(items.begin() + offset, items.end());
            return;
            
            if (end == -1)
            {
                end = (i32)items.size();
            }

            for (i32 i = offset; i < end; i++)
            {
                i32 best = i;

                for (i32 j = i + 1; j < end; j++)
                {
                    //if (items[j].CompareTo(items[best]) > 0)
                    
                    //  https://github.com/liamt19/Lizard/blob/37a61ce5a89fb9b0bb2697762a12b18c06f0f39b/Logic/Data/RootMove.cs#L28
                    if (items[j].Score != items[best].Score) {
                        if (items[j].Score > items[best].Score)
                        {
                            best = j;
                        }
                    }
                    else if (items[j].PreviousScore > items[best].PreviousScore)
                    {
                        best = j;
                    }
                }

                if (best != i)
                {
                    //(items[i], items[best]) = (items[best], items[i]);
                    std::swap(items[i], items[best]);
                }
            }
        }



    }

}










#endif