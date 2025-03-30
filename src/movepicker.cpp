
#include "search.h"
#include "movepicker.h"
#include "movegen.h"
#include "search_options.h"

namespace Horsie::Search {

    Movepicker::Movepicker(Position& pos, HistoryTable& history, SearchStackEntry* _ss, Move ttMove) : 
        Pos(pos),
        History(history),
        ss(_ss),
        TTMove(ttMove)
    {
        
        Stage = MovepickerStage::TT;
        KillerMove = (ss->KillerMove != ttMove) ? ss->KillerMove : Move::Null();

        if (!(ttMove && pos.IsPseudoLegal(ttMove))) {
            ++Stage;
        }
    }


    ScoredMove OrderNext(ScoredMoveList& list, i32 listIndex) {
        i32 max = list[listIndex].score;
        i32 maxIndex = listIndex;

        for (i32 i = listIndex + 1; i < list.Size(); i++) {
            if (list[i].score > max) {
                max = list[i].score;
                maxIndex = i;
            }
        }

        std::swap(list[maxIndex], list[listIndex]);
        return list[listIndex];
    }


    Move Movepicker::Next() {

        switch (Stage) {
        case MovepickerStage::TT:
        {
            ++Stage;
            return TTMove;
        }

        case MovepickerStage::GenNoisies:
        {
            NoisyMoves.GeneratedSize = GenerateNoisy(Pos, NoisyMoves.List, 0);
            ScoreList(NoisyMoves);

            ++Stage;
            [[fallthrough]];
        }
        case MovepickerStage::GoodNoisies:
        {
            while (MoveIndex < NoisyMoves.Size()) {
                const auto sm = OrderNext(NoisyMoves, MoveIndex);
                const auto [move, score] = sm;

                MoveIndex++;

                const auto thresh = -score / 4;
                if (!Pos.SEE_GE(move, thresh)) {
                    BadNoisyMoves.Append(sm);
                }
                else {
                    return move;
                }
            }

            ++Stage;
            [[fallthrough]];
        }

        case MovepickerStage::Killer:
        {
            ++Stage;

            if (!SkipQuiets && KillerMove && Pos.IsPseudoLegal(KillerMove)) {
                return KillerMove;
            }

            [[fallthrough]];
        }

        case MovepickerStage::GenQuiets:
        {
            if (!SkipQuiets) {
                MoveIndex = 0;
                QuietMoves.GeneratedSize = GenerateQuiet(Pos, QuietMoves.List, 0);
                ScoreList(QuietMoves);
            }

            ++Stage;
            [[fallthrough]];
        }
        case MovepickerStage::PlayQuiets:
        {
            if (!SkipQuiets && MoveIndex < QuietMoves.Size()) {
                const auto sm = OrderNext(QuietMoves, MoveIndex);
                const auto [move, score] = sm;
                MoveIndex++;

                return move;
            }

            ++Stage;
            [[fallthrough]];
        }

        case MovepickerStage::StartBadNoisies:
        {
            MoveIndex = 0;

            ++Stage;
            [[fallthrough]];
        }
        case MovepickerStage::BadNoisies:
        {
            if (MoveIndex < BadNoisyMoves.Size()) {
                return BadNoisyMoves[MoveIndex++].move;
            }

            Stage = MovepickerStage::End;
            [[fallthrough]];
        }
        
        case MovepickerStage::End:
        default:
        {
            return Move::Null();
        }

        }

    }


    void Movepicker::ScoreList(ScoredMoveList& list) {
        const auto stm = Pos.ToMove;

        for (i32 i = 0; i < list.Size(); i++) {
            const Move m = list[i].move;
            const auto [moveFrom, moveTo] = m.Unpack();
            const auto type = Pos.Moved(m);

            if (m == TTMove || m == KillerMove) {
                list.RemoveAt(i);
                i--;
                continue;
            }

            if (m.IsPromotion()) {
                list[i].score = PromotionValue(m.PromotionTo());
            }
            else if (Pos.IsCapture(m)) {
                const auto capturedPiece = Pos.Captured(m);
                const auto& hist = History.CaptureHistory[stm][type][moveTo][capturedPiece];
                list[i].score = (MVVMult * GetPieceValue(capturedPiece)) + hist;
            }
            else {
                const auto contIdx = MakePiece(stm, type);

                list[i].score =  2 * History.MainHistory[stm][m.GetMoveMask()];
                list[i].score += 2 * (*(ss - 1)->ContinuationHistory)[contIdx][moveTo];
                list[i].score +=     (*(ss - 2)->ContinuationHistory)[contIdx][moveTo];
                list[i].score +=     (*(ss - 4)->ContinuationHistory)[contIdx][moveTo];
                list[i].score +=     (*(ss - 6)->ContinuationHistory)[contIdx][moveTo];

                if (ss->Ply < LowPlyCount) {
                    list[i].score += ((2 * LowPlyCount + 1) * History.PlyHistory[ss->Ply][m.GetMoveMask()]) / (2 * ss->Ply + 1);
                }

                if (Pos.GivesCheckOnSquare(type, moveTo)) {
                    list[i].score += CheckBonus;
                }
            }

            //if (type == HORSIE) {
            //    list[i].score += 200;
            //}
        }

    }

}