


#include "../nnue/nn.h"

#include "immintrin.h"

#include <fstream>
#include <cstring>

#include "../incbin/incbin.h"

#if !defined(_MSC_VER)
INCBIN(EVAL, EVALFILE);
#else
const unsigned char gEVALData[1] = {};
const unsigned char* const gEVALEnd = &gEVALData[1];
const unsigned int gEVALSize = 1;
#endif

namespace Horsie
{

    namespace NNUE {

        Network net;
        const Network* g_network;

        void LoadNetwork(const std::string& path) {
#if defined(_MSC_VER)
            std::ifstream stream(path, std::ios::binary);

            for (size_t i = 0; i < FeatureWeightElements * InputBuckets; i++)
                net.FeatureWeights[i] = read_little_endian<int16_t>(stream);

            for (size_t i = 0; i < FeatureBiasElements; i++)
                net.FeatureBiases[i] = read_little_endian<int16_t>(stream);

            for (size_t i = 0; i < LayerWeightElements; i++)
                for (size_t bucket = 0; bucket < OutputBuckets; bucket++)
                    net.LayerWeights[bucket][i] = read_little_endian<int16_t>(stream);

            for (size_t i = 0; i < LayerBiasElements; i++)
                net.LayerBiases[i] = read_little_endian<int16_t>(stream);

            g_network = &net;
#else
            const Network* network = reinterpret_cast<const Network*>(gEVALData);

            for (size_t i = 0; i < FeatureWeightElements * InputBuckets; i++)
                net.FeatureWeights[i] = network->FeatureWeights[i];

            for (size_t i = 0; i < FeatureBiasElements; i++)
                net.FeatureBiases[i] = network->FeatureBiases[i];

            const int16_t* buff = reinterpret_cast<const int16_t*>(network->LayerWeights.data()->data());
            size_t indx = 0;
            for (size_t i = 0; i < LayerWeightElements; i++)
                for (size_t bucket = 0; bucket < OutputBuckets; bucket++)
                    net.LayerWeights[bucket][i] = buff[indx++];
                    
            for (size_t i = 0; i < LayerBiasElements; i++) 
                net.LayerBiases[i] = network->LayerBiases[i];

            g_network = &net;
#endif

            
        }


        static constexpr i32 BucketForPerspective(i32 ksq, i32 perspective) {
            return (KingBuckets[(ksq ^ (56 * perspective))]);
        }


        void RefreshAccumulator(Position& pos)
        {
            RefreshAccumulatorPerspectiveFull(pos, WHITE);
            RefreshAccumulatorPerspectiveFull(pos, BLACK);
        }


        void RefreshAccumulatorPerspectiveFull(Position& pos, i32 perspective)
        {
            Accumulator& accumulator = *pos.State->accumulator;
            Bitboard& bb = pos.bb;

            accumulator.Sides[perspective] = g_network->FeatureBiases;
            accumulator.NeedsRefresh[perspective] = false;
            accumulator.Computed[perspective] = true;

            i32 ourKing = pos.State->KingSquares[perspective];
            u64 occ = bb.Occupancy;
            while (occ != 0)
            {
                i32 pieceIdx = poplsb(occ);

                i32 pt = bb.GetPieceAtIndex(pieceIdx);
                i32 pc = bb.GetColorAtIndex(pieceIdx);

                i32 idx = FeatureIndexSingle(pc, pt, pieceIdx, ourKing, perspective);

                const auto accum   = reinterpret_cast<i16*>(&accumulator.Sides[perspective]);
                const auto weights = &g_network->FeatureWeights[idx];
                Add(accum, accum, weights);
            }

            BucketCache& cache = pos.CachedBuckets[BucketForPerspective(ourKing, perspective)];
            Bitboard& entryBB = cache.Boards[perspective];
            Accumulator& entryAcc = cache.accumulator;

            accumulator.CopyTo(&entryAcc, perspective);
            bb.CopyTo(entryBB);
        }


        void RefreshAccumulatorPerspective(Position& pos, i32 perspective)
        {
            Accumulator& accumulator = *pos.State->accumulator;
            Bitboard& bb = pos.bb;

            i32 ourKing = pos.State->KingSquares[perspective];
            i32 thisBucket = KingBuckets[ourKing];

            BucketCache& rtEntry = pos.CachedBuckets[BucketForPerspective(ourKing, perspective)];
            Bitboard& entryBB = rtEntry.Boards[perspective];
            Accumulator& entryAcc = rtEntry.accumulator;

            auto ourAccumulation = reinterpret_cast<i16*>(&entryAcc.Sides[perspective]);
            accumulator.NeedsRefresh[perspective] = false;

            for (i32 pc = 0; pc < COLOR_NB; pc++)
            {
                for (i32 pt = 0; pt < PIECE_NB; pt++)
                {
                    u64 prev = entryBB.Pieces[pt] & entryBB.Colors[pc];
                    u64 curr =      bb.Pieces[pt] &      bb.Colors[pc];

                    u64 added   = curr & ~prev;
                    u64 removed = prev & ~curr;

                    while (added != 0)
                    {
                        i32 sq = poplsb(added);
                        i32 idx = FeatureIndexSingle(pc, pt, sq, ourKing, perspective);

                        const auto weights = &g_network->FeatureWeights[idx];
                        Add(ourAccumulation, ourAccumulation, weights);
                    }

                    while (removed != 0)
                    {
                        i32 sq = poplsb(removed);
                        i32 idx = FeatureIndexSingle(pc, pt, sq, ourKing, perspective);

                        const auto weights = &g_network->FeatureWeights[idx];
                        Sub(ourAccumulation, ourAccumulation, weights);
                    }
                }
            }

            entryAcc.CopyTo(&accumulator, perspective);
            bb.CopyTo(entryBB);

            accumulator.Computed[perspective] = true;
        }


        i32 GetEvaluation(Position& pos) {

            Accumulator& accumulator = *pos.State->accumulator;
            ProcessUpdates(pos);

            Bitboard& bb = pos.bb;

            const __m256i zeroVec = _mm256_setzero_si256();
            const __m256i maxVec = _mm256_set1_epi16(QA);
            __m256i sum = _mm256_setzero_si256();

            i32 occ = (i32)popcount(pos.bb.Occupancy);
            i32 outputBucket = std::min((63 - occ) * (32 - occ) / 225, 7);

            const i32 Stride = (HiddenSize / (sizeof(__m256i) / sizeof(i16))) / 2;

            auto data0 = reinterpret_cast<const __m256i*>(&accumulator.Sides[pos.ToMove]);
            auto data1 = &data0[Stride];
            auto weights = reinterpret_cast<const __m256i*>(&g_network->LayerWeights[outputBucket]);

            for (i32 i = 0; i < Stride; i++) {
                __m256i c_0 = _mm256_min_epi16(maxVec, _mm256_max_epi16(zeroVec, data0[i]));
                __m256i c_1 = _mm256_min_epi16(maxVec, _mm256_max_epi16(zeroVec, data1[i]));
                __m256i mult = _mm256_mullo_epi16(c_0, weights[i]);
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(mult, c_1));
            }

            data0 = reinterpret_cast<const __m256i*>(&accumulator.Sides[Not(pos.ToMove)]);
            data1 = data0 + Stride;
            weights = reinterpret_cast<const __m256i*>(&g_network->LayerWeights[outputBucket][HiddenSize / 2]);
            for (i32 i = 0; i < Stride; i++) {
                __m256i c_0 = _mm256_min_epi16(maxVec, _mm256_max_epi16(zeroVec, data0[i]));
                __m256i c_1 = _mm256_min_epi16(maxVec, _mm256_max_epi16(zeroVec, data1[i]));
                __m256i mult = _mm256_mullo_epi16(c_0, weights[i]);
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(mult, c_1));
            }

            i32 output = hsum_8x32(sum);
            i32 retVal = (output / QA + g_network->LayerBiases[outputBucket]) * OutputScale / QAB;
            //std::cout << "retVal: " << retVal << std::endl;
            return retVal;
        }



        void MakeMoveNN(Position& pos, Move m) {
            Bitboard& bb = pos.bb;

            Accumulator* src = pos.State->accumulator;
            Accumulator* dst = pos.NextState()->accumulator;

            dst->NeedsRefresh[WHITE] = src->NeedsRefresh[WHITE];
            dst->NeedsRefresh[BLACK] = src->NeedsRefresh[BLACK];

            dst->Computed[WHITE] = dst->Computed[BLACK] = false;

            i32 moveTo = m.To();
            i32 moveFrom = m.From();

            i32 us = pos.ToMove;
            i32 ourPiece = bb.GetPieceAtIndex(moveFrom);

            i32 them = Not(us);
            i32 theirPiece = bb.GetPieceAtIndex(moveTo);

            PerspectiveUpdate& wUpdate = dst->Update[WHITE];
            PerspectiveUpdate& bUpdate = dst->Update[BLACK];

            wUpdate.Clear();
            bUpdate.Clear();

            if (ourPiece == KING && (KingBuckets[moveFrom ^ (56 * us)] != KingBuckets[moveTo ^ (56 * us)]))
            {
                //  We will need to fully refresh our perspective, but we can still do theirs.
                dst->NeedsRefresh[us] = true;

                PerspectiveUpdate& theirUpdate = dst->Update[them];
                i32 theirKing = pos.State->KingSquares[them];

                i32 from = FeatureIndexSingle(us, ourPiece, moveFrom, theirKing, them);
                i32 to = FeatureIndexSingle(us, ourPiece, moveTo, theirKing, them);

                if (theirPiece != NONE && !m.IsCastle())
                {
                    i32 cap = FeatureIndexSingle(them, theirPiece, moveTo, theirKing, them);

                    theirUpdate.PushSubSubAdd(from, cap, to);
                }
                else if (m.IsCastle())
                {
                    i32 rookFromSq = moveTo;
                    i32 rookToSq = m.CastlingRookSquare();

                    to = FeatureIndexSingle(us, ourPiece, m.CastlingKingSquare(), theirKing, them);

                    i32 rookFrom = FeatureIndexSingle(us, ROOK, rookFromSq, theirKing, them);
                    i32 rookTo = FeatureIndexSingle(us, ROOK, rookToSq, theirKing, them);

                    theirUpdate.PushSubSubAddAdd(from, rookFrom, to, rookTo);
                }
                else
                {
                    theirUpdate.PushSubAdd(from, to);
                }
            }
            else
            {
                i32 wKing = pos.State->KingSquares[WHITE];
                i32 bKing = pos.State->KingSquares[BLACK];

                const auto [wFrom, bFrom] = FeatureIndex(us, ourPiece, moveFrom, wKing, bKing);
                const auto [wTo, bTo] = FeatureIndex(us, m.IsPromotion() ? m.PromotionTo() : ourPiece, moveTo, wKing, bKing);

                wUpdate.PushSubAdd(wFrom, wTo);
                bUpdate.PushSubAdd(bFrom, bTo);

                if (theirPiece != NONE)
                {
                    const auto [wCap, bCap] = FeatureIndex(them, theirPiece, moveTo, wKing, bKing);

                    wUpdate.PushSub(wCap);
                    bUpdate.PushSub(bCap);
                }
                else if (m.IsEnPassant())
                {
                    i32 idxPawn = moveTo - ShiftUpDir(us);

                    const auto [wCap, bCap] = FeatureIndex(them, PAWN, idxPawn, wKing, bKing);

                    wUpdate.PushSub(wCap);
                    bUpdate.PushSub(bCap);
                }
            }
        }


        void MakeNullMove(Position& pos) {
            Accumulator* currAcc = pos.State->accumulator;
            Accumulator* nextAcc = pos.NextState()->accumulator;

            currAcc->CopyTo(nextAcc);

            nextAcc->Computed[WHITE] = currAcc->Computed[WHITE];
            nextAcc->Computed[BLACK] = currAcc->Computed[BLACK];
            nextAcc->Update[WHITE].Clear();
            nextAcc->Update[BLACK].Clear();
        }


        void ProcessUpdates(Position& pos) {
            StateInfo* st = pos.State;
            for (i32 perspective = 0; perspective < 2; perspective++)
            {
                //  If the current state is correct for our perspective, no work is needed
                if (st->accumulator->Computed[perspective])
                    continue;
                //  If the current state needs a refresh, don't bother with previous states
                if (st->accumulator->NeedsRefresh[perspective])
                {
                    RefreshAccumulatorPerspective(pos, perspective);
                    continue;
                }
                //  Find the most recent computed or refresh-needed accumulator
                StateInfo* curr = st - 1;
                while (!curr->accumulator->Computed[perspective] && !curr->accumulator->NeedsRefresh[perspective])
                    curr--;
                if (curr->accumulator->NeedsRefresh[perspective])
                {
                    //  The most recent accumulator would need to be refreshed,
                    //  so don't bother and refresh the current one instead
                    RefreshAccumulatorPerspective(pos, perspective);
                }
                else
                {
                    //  Update incrementally till the current accumulator is correct
                    while (curr != st)
                    {
                        StateInfo* prev = curr;
                        curr++;
                        UpdateSingle(prev->accumulator, curr->accumulator, perspective);
                    }
                }
            }
        }

        void UpdateSingle(Accumulator* prev, Accumulator* curr, i32 perspective)
        {
            auto FeatureWeights = reinterpret_cast<const i16*>(&g_network->FeatureWeights[0]);
            const auto& updates = curr->Update[perspective];

            if (updates.AddCnt == 0 && updates.SubCnt == 0)
            {
                //  For null moves, we still need to carry forward the correct accumulator state
                prev->CopyTo(curr, perspective);
                return;
            }

            auto src = reinterpret_cast<i16*>(&prev->Sides[perspective]);
            auto dst = reinterpret_cast<i16*>(&curr->Sides[perspective]);
            if (updates.AddCnt == 1 && updates.SubCnt == 1)
            {
                SubAdd(src, dst,
                    (FeatureWeights + updates.Subs[0]),
                    (FeatureWeights + updates.Adds[0]));
            }
            else if (updates.AddCnt == 1 && updates.SubCnt == 2)
            {
                SubSubAdd(src, dst,
                    (FeatureWeights + updates.Subs[0]),
                    (FeatureWeights + updates.Subs[1]),
                    (FeatureWeights + updates.Adds[0]));
            }
            else if (updates.AddCnt == 2 && updates.SubCnt == 2)
            {
                SubSubAddAdd(src, dst,
                    (FeatureWeights + updates.Subs[0]),
                    (FeatureWeights + updates.Subs[1]),
                    (FeatureWeights + updates.Adds[0]),
                    (FeatureWeights + updates.Adds[1]));
            }
            curr->Computed[perspective] = true;
        }


        std::pair<i32, i32> FeatureIndex(i32 pc, i32 pt, i32 sq, i32 wk, i32 bk)
        {
            const i32 ColorStride = 64 * 6;
            const i32 PieceStride = 64;

            i32 wSq = sq;
            i32 bSq = sq ^ 56;

            if (wk % 8 > 3)
            {
                wk ^= 7;
                wSq ^= 7;
            }

            bk ^= 56;
            if (bk % 8 > 3)
            {
                bk ^= 7;
                bSq ^= 7;
            }

            i32 whiteIndex = (768 * KingBuckets[wk]) + (pc * ColorStride) + (pt * PieceStride) + wSq;
            i32 blackIndex = (768 * KingBuckets[bk]) + (Not(pc) * ColorStride) + (pt * PieceStride) + bSq;

            return { whiteIndex * HiddenSize, blackIndex * HiddenSize };
        }

        i32 FeatureIndexSingle(i32 pc, i32 pt, i32 sq, i32 kingSq, i32 perspective)
        {
            const i32 ColorStride = 64 * 6;
            const i32 PieceStride = 64;

            if (perspective == BLACK)
            {
                sq ^= 56;
                kingSq ^= 56;
            }

            if (kingSq % 8 > 3)
            {
                sq ^= 7;
                kingSq ^= 7;
            }

            return ((768 * KingBuckets[kingSq]) + ((pc ^ perspective) * ColorStride) + (pt * PieceStride) + (sq)) * HiddenSize;
        }


        void ResetCaches(Position& pos) {
            for (auto& bucket : pos.CachedBuckets) {
                bucket.accumulator.Sides[WHITE] = bucket.accumulator.Sides[BLACK] = g_network->FeatureBiases;
                bucket.Boards[WHITE].Reset();
                bucket.Boards[BLACK].Reset();
            }
        }


        //  https://stackoverflow.com/questions/63106143/simd-c-avx2-intrinsics-getting-sum-of-m256i-vector-with-16bit-integers
        static int32_t hsum_8x32(__m256i v) {
            // silly GCC uses a longer AXV512VL instruction if AVX512 is enabled :/
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));

            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);                  // 3-operand non-destructive AVX lets us save a u8 without needing a movdqa
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));    // Swap the low two elements
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            return _mm_cvtsi128_si32(sum32);       // movd
        }

        static void SubSubAddAdd(const i16* _src, i16* _dst, const i16* _sub1, const i16* _sub2, const i16* _add1, const i16* _add2) {
            const __m256i* src  = reinterpret_cast<const __m256i*>(_src);
                  __m256i* dst  = reinterpret_cast<      __m256i*>(_dst);
            const __m256i* sub1 = reinterpret_cast<const __m256i*>(_sub1);
            const __m256i* sub2 = reinterpret_cast<const __m256i*>(_sub2);
            const __m256i* add1 = reinterpret_cast<const __m256i*>(_add1);
            const __m256i* add2 = reinterpret_cast<const __m256i*>(_add2);
            for (i32 i = 0; i < SIMD_CHUNKS; i++)
            {
                dst[i] = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(src[i], add1[i]), add2[i]), sub1[i]), sub2[i]);
            }
        }

        static void SubSubAdd(const i16* _src, i16* _dst, const i16* _sub1, const i16* _sub2, const i16* _add1) {
            const __m256i* src  = reinterpret_cast<const __m256i*>(_src);
                  __m256i* dst  = reinterpret_cast<      __m256i*>(_dst);
            const __m256i* sub1 = reinterpret_cast<const __m256i*>(_sub1);
            const __m256i* sub2 = reinterpret_cast<const __m256i*>(_sub2);
            const __m256i* add1 = reinterpret_cast<const __m256i*>(_add1);
            for (i32 i = 0; i < SIMD_CHUNKS; i++)
            {
                dst[i] = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(src[i], add1[i]), sub1[i]), sub2[i]);
            }
        }

        static void SubAdd(const i16* _src, i16* _dst, const i16* _sub1, const i16* _add1) {
            const __m256i* src  = reinterpret_cast<const __m256i*>(_src);
                  __m256i* dst  = reinterpret_cast<      __m256i*>(_dst);
            const __m256i* sub1 = reinterpret_cast<const __m256i*>(_sub1);
            const __m256i* add1 = reinterpret_cast<const __m256i*>(_add1);
            for (i32 i = 0; i < SIMD_CHUNKS; i++)
            {
                dst[i] = _mm256_sub_epi16(_mm256_add_epi16(src[i], add1[i]), sub1[i]);
            }
        }

        static void Add(const i16* _src, i16* _dst, const i16* _add1) {
            const __m256i* src  = reinterpret_cast<const __m256i*>(_src);
                  __m256i* dst  = reinterpret_cast<      __m256i*>(_dst);
            const __m256i* add1 = reinterpret_cast<const __m256i*>(_add1);
            for (i32 i = 0; i < SIMD_CHUNKS; i++)
                dst[i] = _mm256_add_epi16(src[i], add1[i]);
        }

        static void Sub(const i16* _src, i16* _dst, const i16* _sub1) {
            const __m256i* src  = reinterpret_cast<const __m256i*>(_src);
                  __m256i* dst  = reinterpret_cast<      __m256i*>(_dst);
            const __m256i* sub1 = reinterpret_cast<const __m256i*>(_sub1);
            for (i32 i = 0; i < SIMD_CHUNKS; i++)
                dst[i] = _mm256_sub_epi16(src[i], sub1[i]);
        }
    }

}
