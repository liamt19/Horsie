#pragma once


#ifndef SEARCH_H
#define SEARCH_H

#include <vector>
#include <chrono>

#include "move.h"
#include "history.h"

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
            int MaxDepth = Horsie::MaxDepth;
            ulong MaxNodes = UINT64_MAX;
        };

        struct SearchStackEntry {
        public:
            Move CurrentMove;
            Move Skip;
            PieceToHistory* ContinuationHistory;
            int StatScore;
            int DoubleExtensions;
            short Ply;
            short StaticEval;
            bool InCheck;
            bool TTPV;
            bool TTHit;
            //uint8_t _pad0[1];
            Move* PV;
            //uint8_t _pad1[8];
            Move Killer0;
            //uint8_t _pad3[4];
            Move Killer1;
            //uint8_t _pad4[4];

            void Clear() {
                CurrentMove = Skip = Killer0 = Killer1 = Move::Null();
                ContinuationHistory = nullptr;

                StatScore = 0;
                Ply = 0;
                DoubleExtensions = 0;
                StaticEval = ScoreNone;

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
                Score = PreviousScore = AverageScore = -ScoreInfinite;
                Depth = 0;
                PVLength = 1;
            }
            bool operator==(const Move& m) const { return PV[0] == m; }
            bool operator<(const RootMove& m) const { return m.Score != Score ? m.Score < Score : m.PreviousScore < PreviousScore; }
            
            Move Move;
            int Score;
            int PreviousScore;
            int AverageScore;
            int Depth;
            int PVLength;

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

            std::chrono::steady_clock::time_point TimeStart = std::chrono::high_resolution_clock::now();
            ulong SearchTimeMS = 0;
            ulong MaxNodes = 0;
            bool StopSearching = false;


            std::vector<RootMove> RootMoves;
            HistoryTable History;
            ulong NodeTable[64][64];

            Move CurrentMove() const { return RootMoves[PVIndex].Move; }


            void Search(Position& pos);

            template <SearchNodeType NodeType>
            int Negamax(Position& pos, SearchStackEntry* ss, int alpha, int beta, int depth, bool cutNode);

            template <SearchNodeType NodeType>
            int QSearch(Position& pos, SearchStackEntry* ss, int alpha, int beta, int depth);



            inline int StatBonus(int depth) const { return std::min((StatBonusMult * depth) - StatBonusSub, StatBonusMax); }
            inline int StatMalus(int depth) const { return std::min((StatMalusMult * depth) - StatMalusSub, StatMalusMax); }

            void AssignScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, int size, Move ttMove);
            Move OrderNextMove(ScoredMove* moves, int size, int listIndex);

            void UpdatePV(Move* pv, Move move, Move* childPV);

            void UpdateContinuations(SearchStackEntry* ss, int pc, int pt, int sq, int bonus);
            void UpdateStats(Position pos, SearchStackEntry* ss, Move bestMove, int bestScore, int beta, int depth, Move* quietMoves, int quietCount, Move* captureMoves, int captureCount);

            bool SEE_GE(Position pos, Move m, int threshold = 1);



            bool CheckTime() const {

                if (Nodes >= MaxNodes)
                    return true;
                
                auto now = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - TimeStart);

                return (duration.count() > SearchTimeMS);
            }

            void Reset() {
                Nodes = 0;
                PVIndex = RootDepth = SelDepth = CompletedDepth = CheckupCount = 0;
            }

        private:
            const int CheckupMax = 512;
        };


        static void StableSort(std::vector<RootMove> items, int offset = 0, int end = -1) {
            
            if (end == -1)
            {
                end = items.size();
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