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

    static const i32 DepthQChecks = 0;
    static const i32 DepthQNoChecks = -1;

    static const i32 SEARCH_TIME_BUFFER = 25;

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



        class SearchThread
        {
        public:
            u64 Nodes;
            i32 NMPPly;
            i32 ThreadIdx;
            i32 PVIndex;
            i32 RootDepth;
            i32 SelDepth;
            i32 CompletedDepth;
            i32 CheckupCount = 0;
            bool Searching;
            bool Quit;
            bool IsMain = false;

            std::chrono::system_clock::time_point TimeStart = std::chrono::system_clock::now();
            i32 SearchTimeMS = 0;
            u64 MaxNodes = 0;
            bool StopSearching = false;

            std::function<void()> OnDepthFinish;
            std::vector<RootMove> RootMoves{};
            HistoryTable History{};

            bool HasSoftTime = false;
            i32 SoftTimeLimit = 0;
            Util::NDArray<u64, 64, 64> NodeTable{};

            Move CurrentMove() const { return RootMoves[PVIndex].move; }


            void Search(Position& pos, SearchLimits& info);

            template <SearchNodeType NodeType>
            i32 Negamax(Position& pos, SearchStackEntry* ss, i32 alpha, i32 beta, i32 depth, bool cutNode);

            template <SearchNodeType NodeType>
            i32 QSearch(Position& pos, SearchStackEntry* ss, i32 alpha, i32 beta);

            void UpdateCorrectionHistory(Position& pos, i32 diff, i32 depth);
            i16 AdjustEval(Position& pos, i32 us, i16 rawEval);

            inline i32 GetRFPMargin(i32 depth, bool improving) const { return (depth - (improving)) * RFPMargin; }
            inline i32 StatBonus(i32 depth) const { return std::min((StatBonusMult * depth) - StatBonusSub, StatBonusMax); }
            inline i32 StatMalus(i32 depth) const { return std::min((StatMalusMult * depth) - StatMalusSub, StatMalusMax); }

            void AssignProbCutScores(Position& pos, ScoredMove* list, i32 size);
            void AssignQuiescenceScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, i32 size, Move ttMove);
            void AssignScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, i32 size, Move ttMove);
            Move OrderNextMove(ScoredMove* moves, i32 size, i32 listIndex);

            void UpdatePV(Move* pv, Move move, Move* childPV);

            void UpdateContinuations(SearchStackEntry* ss, i32 pc, i32 pt, i32 sq, i32 bonus);
            void UpdateStats(Position& pos, SearchStackEntry* ss, Move bestMove, i32 bestScore, i32 beta, i32 depth, Move* quietMoves, i32 quietCount, Move* captureMoves, i32 captureCount);

            std::string Debug_GetMovesPlayed(SearchStackEntry* ss);



            i64 GetSearchTime() const {
                auto now = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - TimeStart);
                return duration.count();
            }

            bool CheckTime() const {
                const auto time = GetSearchTime();
                return (time > SearchTimeMS - SEARCH_TIME_BUFFER);
            }

            void MakeMoveTime(SearchLimits& limits) {
                i32 newSearchTime = limits.Increment + std::max(limits.PlayerTime / 2, limits.PlayerTime / limits.MovesToGo);

                newSearchTime = std::min(newSearchTime, limits.PlayerTime);

                //  Values from Clarity
                SoftTimeLimit = static_cast<i32>(0.65 * ((static_cast<double>(limits.PlayerTime) / limits.MovesToGo) + (limits.Increment * 3 / 4.0)));
                HasSoftTime = true;

                limits.MaxSearchTime = newSearchTime;
            }

            void Reset() {
                Nodes = 0;
                PVIndex = RootDepth = SelDepth = CompletedDepth = CheckupCount = NMPPly = 0;
                SoftTimeLimit = 0;
                HasSoftTime = StopSearching = false;
            }

        private:
            const i32 CheckupMax = 512;
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