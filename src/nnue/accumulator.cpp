
#include "accumulator.h"

#include "nn.h"
#include "../bitboard.h"
#include "../move.h"
#include "../position.h"

#include <cassert>

namespace Horsie::NNUE {

    void AccumulatorStack::MoveNext() {
#if defined(NO)
        if (++HeadIndex == AccStack.size()) {
            AccStack.emplace_back();
        }

        CurrentAccumulator = &AccStack[HeadIndex];
#endif
    }

    void AccumulatorStack::UndoMove() {
#if defined(NO)
        assert(HeadIndex > 0);
        HeadIndex--;

        CurrentAccumulator = &AccStack[HeadIndex];
#endif
    }

    void AccumulatorStack::MakeMove(const Position& pos, Move m) {
#if defined(NO)
        const auto& bb = pos.bb;

        MoveNext();

        Accumulator* src = &AccStack[HeadIndex - 1];
        Accumulator* dst = &AccStack[HeadIndex];

        dst->NeedsRefresh[WHITE] = src->NeedsRefresh[WHITE];
        dst->NeedsRefresh[BLACK] = src->NeedsRefresh[BLACK];

        dst->Computed[WHITE] = dst->Computed[BLACK] = false;

        const auto [moveFrom, moveTo] = m.Unpack();

        const auto us = pos.ToMove;
        const auto ourPiece = bb.GetPieceAtIndex(moveFrom);

        const auto them = Not(us);
        const auto theirPiece = bb.GetPieceAtIndex(moveTo);

        PerspectiveUpdate& wUpdate = dst->Update[WHITE];
        PerspectiveUpdate& bUpdate = dst->Update[BLACK];

        wUpdate.Clear();
        bUpdate.Clear();

        if (ourPiece == KING && (KingBuckets[moveFrom ^ (56 * us)] != KingBuckets[moveTo ^ (56 * us)])) {
            //  We will need to fully refresh our perspective, but we can still do theirs.
            dst->NeedsRefresh[us] = true;

            PerspectiveUpdate& theirUpdate = dst->Update[them];
            i32 theirKing = pos.KingSquare(them);

            i32 from = FeatureIndexSingle(us, ourPiece, moveFrom, theirKing, them);
            i32 to = FeatureIndexSingle(us, ourPiece, moveTo, theirKing, them);

            if (theirPiece != NONE && !m.IsCastle()) {
                i32 cap = FeatureIndexSingle(them, theirPiece, moveTo, theirKing, them);

                theirUpdate.PushSubSubAdd(from, cap, to);
            }
            else if (m.IsCastle()) {
                i32 rookFromSq = moveTo;
                i32 rookToSq = m.CastlingRookSquare();

                to = FeatureIndexSingle(us, ourPiece, m.CastlingKingSquare(), theirKing, them);

                i32 rookFrom = FeatureIndexSingle(us, ROOK, rookFromSq, theirKing, them);
                i32 rookTo = FeatureIndexSingle(us, ROOK, rookToSq, theirKing, them);

                theirUpdate.PushSubSubAddAdd(from, rookFrom, to, rookTo);
            }
            else {
                theirUpdate.PushSubAdd(from, to);
            }
        }
        else {
            i32 wKing = pos.KingSquare(WHITE);
            i32 bKing = pos.KingSquare(BLACK);

            const auto [wFrom, bFrom] = FeatureIndex(us, ourPiece, moveFrom, wKing, bKing);
            const auto [wTo, bTo] = FeatureIndex(us, m.IsPromotion() ? m.PromotionTo() : ourPiece, moveTo, wKing, bKing);

            wUpdate.PushSubAdd(wFrom, wTo);
            bUpdate.PushSubAdd(bFrom, bTo);

            if (theirPiece != NONE) {
                const auto [wCap, bCap] = FeatureIndex(them, theirPiece, moveTo, wKing, bKing);

                wUpdate.PushSub(wCap);
                bUpdate.PushSub(bCap);
            }
            else if (m.IsEnPassant()) {
                i32 idxPawn = moveTo - ShiftUpDir(us);

                const auto [wCap, bCap] = FeatureIndex(them, PAWN, idxPawn, wKing, bKing);

                wUpdate.PushSub(wCap);
                bUpdate.PushSub(bCap);
            }
        }
#endif
    }

    void AccumulatorStack::RefreshIntoCache(Position& pos) {
        RefreshPosition(pos);
    }

    void AccumulatorStack::RefreshPosition(Position& pos) {
#if defined(NO)
        i32 perspective = pos.ToMove;
        auto accumulator = CurrentAccumulator;
        Bitboard& bb = pos.bb;

        std::memcpy(&accumulator->Accumulation, &net.Conv1Weights, sizeof(accumulator->Accumulation));
        accumulator->NeedsRefresh[perspective] = false;
        accumulator->Computed[perspective] = true;

        i32 ourKing = pos.KingSquare(perspective);
        u64 occ = bb.Occupancy;
        while (occ != 0) {
            i32 pieceIdx = poplsb(occ);

            i32 pt = bb.GetPieceAtIndex(pieceIdx);
            i32 pc = bb.GetColorAtIndex(pieceIdx);

            i32 idx = FeatureIndexSingle(pc, pt, pieceIdx, ourKing, perspective);

            const auto accum = reinterpret_cast<i16*>(&accumulator->Accumulation);
            const auto weights = accum;// &net.FTWeights[idx];
            Add(accum, accum, weights);
        }
#endif
    }

}
