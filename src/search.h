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

#include "search_options.h"

namespace Horsie {

    static const int DepthQChecks = 0;
    static const int DepthQNoChecks = -1;

    static const int SEARCH_TIME_BUFFER = 25;

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
            int MaxDepth = Horsie::MaxDepth;
            ulong MaxNodes = UINT64_MAX;
            ulong SoftNodeLimit = UINT64_MAX;
            int MaxSearchTime = INT32_MAX;
            int Increment = 0;
            int MovesToGo = 20;

            int MoveTime = 0;
            const bool HasMoveTime() const { return MoveTime != 0; }

            int PlayerTime = 0;
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
            short DoubleExtensions;
            short Ply;
            short StaticEval;
            Move KillerMove;
            Move CurrentMove;
            Move Skip;
            bool InCheck;
            bool TTPV;
            bool TTHit;
            int PVLength;


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
            int Score;
            int PreviousScore;
            int AverageScore;
            int Depth;

            std::vector<Horsie::Move> PV;
        };



        class SearchThread
        {
        public:
            ulong Nodes;
            int ThreadIdx;
            int PVIndex;
            int RootDepth;
            int SelDepth;
            int CompletedDepth;
            int CheckupCount = 0;
            bool Searching;
            bool Quit;
            bool IsMain = false;

            std::chrono::system_clock::time_point TimeStart = std::chrono::system_clock::now();
            int SearchTimeMS = 0;
            ulong MaxNodes = 0;
            bool StopSearching = false;

            std::function<void()> OnDepthFinish;
            std::vector<RootMove> RootMoves{};
            HistoryTable History{};

            bool HasSoftTime = false;
            int SoftTimeLimit = 0;
            Util::NDArray<ulong, 64, 64> NodeTable{};

            Move CurrentMove() const { return RootMoves[PVIndex].move; }


            void Search(Position& pos, SearchLimits& info);

            template <SearchNodeType NodeType>
            int Negamax(Position& pos, SearchStackEntry* ss, int alpha, int beta, int depth, bool cutNode);

            template <SearchNodeType NodeType>
            int QSearch(Position& pos, SearchStackEntry* ss, int alpha, int beta);

            void UpdateCorrectionHistory(Position& pos, int diff, int depth);
            short AdjustEval(Position& pos, int us, short rawEval);

            inline int GetRFPMargin(int depth, bool improving) const { return (depth - (improving)) * RFPMargin; }
            inline int StatBonus(int depth) const { return std::min((StatBonusMult * depth) - StatBonusSub, StatBonusMax); }
            inline int StatMalus(int depth) const { return std::min((StatMalusMult * depth) - StatMalusSub, StatMalusMax); }

            void AssignProbCutScores(Position& pos, ScoredMove* list, int size);
            void AssignQuiescenceScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, int size, Move ttMove);
            void AssignScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, int size, Move ttMove);
            Move OrderNextMove(ScoredMove* moves, int size, int listIndex);

            void UpdatePV(Move* pv, Move move, Move* childPV);

            void UpdateContinuations(SearchStackEntry* ss, int pc, int pt, int sq, int bonus);
            void UpdateStats(Position& pos, SearchStackEntry* ss, Move bestMove, int bestScore, int beta, int depth, Move* quietMoves, int quietCount, Move* captureMoves, int captureCount);

            std::string Debug_GetMovesPlayed(SearchStackEntry* ss);



            long long GetSearchTime() const {
                auto now = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - TimeStart);
                return duration.count();
            }

            bool CheckTime() const {
                //auto now = std::chrono::system_clock::now();
                //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - TimeStart);
                //return (duration.count() > SearchTimeMS);

                const auto time = GetSearchTime();
                return (time > SearchTimeMS - SEARCH_TIME_BUFFER);
            }

            void MakeMoveTime(SearchLimits& limits) {
                int newSearchTime = limits.Increment + (limits.PlayerTime / 2);

                if (limits.MovesToGo != -1)
                {
                    newSearchTime = std::max(newSearchTime, limits.Increment + (limits.PlayerTime / limits.MovesToGo));
                }

                if (newSearchTime > limits.PlayerTime)
                {
                    newSearchTime = limits.PlayerTime;
                }

                //  Values from Clarity
                SoftTimeLimit = (int)(0.6 * ((static_cast<double>(limits.PlayerTime) / limits.MovesToGo) + (limits.Increment * 3 / 4.0)));
                HasSoftTime = true;

                limits.MaxSearchTime = newSearchTime;
            }

            void Reset() {
                Nodes = 0;
                PVIndex = RootDepth = SelDepth = CompletedDepth = CheckupCount = 0;
                SoftTimeLimit = 0;
                HasSoftTime = StopSearching = false;
            }

        private:
            const int CheckupMax = 512;
        };


        static void StableSort(std::vector<RootMove>& items, int offset = 0, int end = -1) {
            
            std::stable_sort(items.begin() + offset, items.end());
            return;
            
            if (end == -1)
            {
                end = (int)items.size();
            }

            for (int i = offset; i < end; i++)
            {
                int best = i;

                for (int j = i + 1; j < end; j++)
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