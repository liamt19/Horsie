
#include "accumulator.h"

#include "nn.h"
#include "../bitboard.h"
#include "../move.h"
#include "../position.h"

#include <cassert>

namespace Horsie::NNUE {

    void AccumulatorStack::MoveNext() {
        if (++HeadIndex == AccStack.size()) {
            AccStack.emplace_back();
        }

        CurrentAccumulator = &AccStack[HeadIndex];
    }

    void AccumulatorStack::UndoMove() {
        assert(HeadIndex > 0);
        HeadIndex--;

        CurrentAccumulator = &AccStack[HeadIndex];
    }

    void AccumulatorStack::MakeMove(const Position& pos, Move m) {
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
    }


    void AccumulatorStack::EnsureUpdated(Position& pos) {

        for (i32 perspective = 0; perspective < 2; perspective++) {
            //  If the current state is correct for our perspective, no work is needed
            if (CurrentAccumulator->Computed[perspective])
                continue;

            //  If the current state needs a refresh, don't bother with previous states
            if (CurrentAccumulator->NeedsRefresh[perspective]) {
                RefreshFromCache(pos, perspective);
                continue;
            }

            //  Find the most recent computed or refresh-needed accumulator
            Accumulator* curr = &AccStack[HeadIndex - 1];
            while (!curr->Computed[perspective] && !curr->NeedsRefresh[perspective])
                curr--;

            if (curr->NeedsRefresh[perspective]) {
                //  The most recent accumulator would need to be refreshed,
                //  so don't bother and refresh the current one instead
                RefreshFromCache(pos, perspective);
            }
            else {
                //  Update incrementally till the current accumulator is correct
                while (curr != CurrentAccumulator) {
                    Accumulator* prev = curr;
                    curr++;
                    ProcessUpdate(prev, curr, perspective);
                }
            }
        }
    }

    void AccumulatorStack::ProcessUpdate(Accumulator* prev, Accumulator* curr, i32 perspective) {
        auto FeatureWeights = reinterpret_cast<const i16*>(&net.FTWeights[0]);
        const auto& updates = curr->Update[perspective];

        assert(updates.AddCnt != 0 || updates.SubCnt != 0);

        auto src = reinterpret_cast<i16*>(&prev->Sides[perspective]);
        auto dst = reinterpret_cast<i16*>(&curr->Sides[perspective]);
        if (updates.AddCnt == 1 && updates.SubCnt == 1) {
            SubAdd(src, dst,
                   &FeatureWeights[updates.Subs[0]],
                   &FeatureWeights[updates.Adds[0]]);
        }
        else if (updates.AddCnt == 1 && updates.SubCnt == 2) {
            SubSubAdd(src, dst,
                      &FeatureWeights[updates.Subs[0]],
                      &FeatureWeights[updates.Subs[1]],
                      &FeatureWeights[updates.Adds[0]]);
        }
        else if (updates.AddCnt == 2 && updates.SubCnt == 2) {
            SubSubAddAdd(src, dst,
                         &FeatureWeights[updates.Subs[0]],
                         &FeatureWeights[updates.Subs[1]],
                         &FeatureWeights[updates.Adds[0]],
                         &FeatureWeights[updates.Adds[1]]);
        }
        curr->Computed[perspective] = true;
    }


    void AccumulatorStack::RefreshIntoCache(Position& pos) {
        RefreshIntoCache(pos, WHITE);
        RefreshIntoCache(pos, BLACK);
    }

    void AccumulatorStack::RefreshIntoCache(Position& pos, i32 perspective) {
        auto accumulator = CurrentAccumulator;
        Bitboard& bb = pos.bb;

        accumulator->Sides[perspective] = net.FTBiases;
        accumulator->NeedsRefresh[perspective] = false;
        accumulator->Computed[perspective] = true;

        i32 ourKing = pos.KingSquare(perspective);
        u64 occ = bb.Occupancy;
        while (occ != 0) {
            i32 pieceIdx = poplsb(occ);

            i32 pt = bb.GetPieceAtIndex(pieceIdx);
            i32 pc = bb.GetColorAtIndex(pieceIdx);

            i32 idx = FeatureIndexSingle(pc, pt, pieceIdx, ourKing, perspective);

            const auto accum = reinterpret_cast<i16*>(&accumulator->Sides[perspective]);
            const auto weights = &net.FTWeights[idx];
            Add(accum, accum, weights);
        }

        auto& cache = pos.CachedBuckets[BucketForPerspective(ourKing, perspective)];
        auto& entryBB = cache.Boards[perspective];
        auto& entryAcc = cache.accumulator;

        accumulator->CopyTo(entryAcc, perspective);
        bb.CopyTo(entryBB);
    }

    void AccumulatorStack::RefreshFromCache(Position& pos, i32 perspective) {
        auto accumulator = CurrentAccumulator;
        Bitboard& bb = pos.bb;

        i32 ourKing = pos.KingSquare(perspective);

        auto& rtEntry = pos.CachedBuckets[BucketForPerspective(ourKing, perspective)];
        auto& entryBB = rtEntry.Boards[perspective];
        auto& entryAcc = rtEntry.accumulator;

        auto ourAccumulation = reinterpret_cast<i16*>(&entryAcc.Sides[perspective]);
        accumulator->NeedsRefresh[perspective] = false;

        for (i32 pc = 0; pc < COLOR_NB; pc++) {
            for (i32 pt = 0; pt < PIECE_NB; pt++) {
                u64 prev = entryBB.Pieces[pt] & entryBB.Colors[pc];
                u64 curr =      bb.Pieces[pt] &      bb.Colors[pc];

                u64 added   = curr & ~prev;
                u64 removed = prev & ~curr;

                while (added != 0) {
                    i32 sq = poplsb(added);
                    i32 idx = FeatureIndexSingle(pc, pt, sq, ourKing, perspective);

                    const auto weights = &net.FTWeights[idx];
                    Add(ourAccumulation, ourAccumulation, weights);
                }

                while (removed != 0) {
                    i32 sq = poplsb(removed);
                    i32 idx = FeatureIndexSingle(pc, pt, sq, ourKing, perspective);

                    const auto weights = &net.FTWeights[idx];
                    Sub(ourAccumulation, ourAccumulation, weights);
                }
            }
        }

        entryAcc.CopyTo(accumulator, perspective);
        bb.CopyTo(entryBB);

        accumulator->Computed[perspective] = true;
    }

}
