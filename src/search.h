#pragma once


#ifndef SEARCH_H
#define SEARCH_H

#include <vector>

#include "move.h"
#include "history.h"

namespace Horsie {

    enum SearchNodeType {
        PVNode,
        NonPVNode,
        RootNode
    };

    class TranspositionTable;
    class Position;

    namespace Search {

        constexpr short ScoreNone = 32760;
        constexpr int ScoreInfinite = 31200;
        constexpr int ScoreMate = 30000;
        constexpr int ScoreDraw = 0;
        constexpr int ScoreTTWin = ScoreMate - 512;
        constexpr int ScoreTTLoss = -ScoreTTWin;
        constexpr int ScoreMateMax = ScoreMate - 256;
        constexpr int ScoreMatedMax = -ScoreMateMax;
        constexpr int ScoreAssuredWin = 20000;
        constexpr int ScoreWin = 10000;


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

        };

        struct RootMove {

            explicit RootMove(Move m) : pv(1, m) {}
            bool operator==(const Move& m) const { return pv[0] == m; }
            bool operator<(const RootMove& m) const { return m.Score != Score ? m.Score < Score : m.PreviousScore < PreviousScore; }

            Move Move;
            int Score;
            int PreviousScore;
            int AverageScore;
            int Depth;

            std::vector<Horsie::Move> pv;
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

            std::vector<RootMove> RootMoves;
            HistoryTable History;
            ulong NodeTable[64][64];

            Move CurrentMove() const { return RootMoves[PVIndex].Move; }
        };



        template <SearchNodeType NodeType>
        int Negamax(Position& pos, SearchStackEntry* ss, int alpha, int beta, int depth, bool cutNode);

        template <SearchNodeType NodeType>
        int QSearch(Position& pos, SearchStackEntry* ss, int alpha, int beta, int depth, bool cutNode);
    }

}










#endif