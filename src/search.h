#pragma once

#include "defs.h"
#include "history.h"
#include "move.h"
#include "util.h"
#include "util/alloc.h"

#include <cstdint>
#include <vector>

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
            i32 PlayerTime = 0;

            constexpr bool HasMoveTime() const { return MoveTime != 0; }
            constexpr bool HasPlayerTime() const { return PlayerTime != 0; }
            constexpr bool IsInfinite() const { return MaxDepth == Horsie::MaxDepth && MaxSearchTime == INT32_MAX; }

            i32 SetTimeLimits() {
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
        };

        struct alignas(64) SearchStackEntry {
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
                if (PV != nullptr) {
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
                Nodes = 0;
            }
            bool operator==(const Move& m) const { return PV[0] == m; }
            bool operator<(const RootMove& m) const { return m.Score != Score ? m.Score < Score : m.PreviousScore < PreviousScore; }
            
            Move move;
            i32 Score;
            i32 PreviousScore;
            i32 AverageScore;
            i32 Depth;
            u64 Nodes;

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

    }
}
