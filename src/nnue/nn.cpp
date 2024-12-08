
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

#if defined(PERM_COUNT)
        std::array<u64, L1_SIZE> NNZCounts = {};
        u64 ActivationCount = 0;
        u64 EvalCalls = 0;
#endif

        NNZTable nnzTable;
        Network net;
        const Network* g_network;

        void LoadNetwork(const std::string& path) {
            SetupNNZ();

#if defined(_MSC_VER)
            std::ifstream stream(path, std::ios::binary);

            std::unique_ptr<QuantisedNetwork> UQNet = std::make_unique<QuantisedNetwork>();

            for (size_t i = 0; i < INPUT_SIZE * L1_SIZE * INPUT_BUCKETS; i++)
                UQNet->FTWeights[i] = read_little_endian<int16_t>(stream);

            for (size_t i = 0; i < L1_SIZE; i++)
                UQNet->FTBiases[i] = read_little_endian<int16_t>(stream);
                
            i8* l1Ptr = reinterpret_cast<i8*>(UQNet->L1Weights);
            for (size_t i = 0; i < L1_SIZE * L2_SIZE * OUTPUT_BUCKETS; i++)
                l1Ptr[i] = read_little_endian<i8>(stream);

            float* ptr = reinterpret_cast<float*>(UQNet->L1Biases);
            for (size_t i = 0; i < L2_SIZE * OUTPUT_BUCKETS; i++)
                ptr[i] = read_little_endian<float>(stream);

            ptr = reinterpret_cast<float*>(UQNet->L2Weights);
            for (size_t i = 0; i < L2_SIZE * L3_SIZE * OUTPUT_BUCKETS; i++)
                ptr[i] = read_little_endian<float>(stream);

            ptr = reinterpret_cast<float*>(UQNet->L2Biases);
            for (size_t i = 0; i < L3_SIZE * OUTPUT_BUCKETS; i++)
                ptr[i] = read_little_endian<float>(stream);

            ptr = reinterpret_cast<float*>(UQNet->L3Weights);
            for (size_t i = 0; i < L3_SIZE * OUTPUT_BUCKETS; i++)
                ptr[i] = read_little_endian<float>(stream);

            ptr = reinterpret_cast<float*>(UQNet->L3Biases);
            for (size_t i = 0; i < OUTPUT_BUCKETS; i++)
                ptr[i] = read_little_endian<float>(stream);

            auto& tempL1 = UQNet->L1Weights;
#else
            const QuantisedNetwork* UQNet = reinterpret_cast<const QuantisedNetwork*>(gEVALData);
            auto tempL1 = new i8[L1_SIZE][OUTPUT_BUCKETS][L2_SIZE];
            std::memcpy(tempL1, UQNet->L1Weights, L1_SIZE * OUTPUT_BUCKETS * L2_SIZE);
#endif

            for (size_t i = 0; i < INPUT_SIZE * L1_SIZE * INPUT_BUCKETS; i++)
                net.FTWeights[i] = UQNet->FTWeights[i];

            for (size_t i = 0; i < L1_SIZE; i++)
                net.FTBiases[i] = UQNet->FTBiases[i];

            PermuteFT(Span<i16>(net.FTWeights), Span<i16>(net.FTBiases));
            PermuteL1(tempL1);

            for (i32 bucket = 0; bucket < OUTPUT_BUCKETS; bucket++)
            {
                for (i32 i = 0; i < L1_SIZE / L1_CHUNK_PER_32; ++i)
                    for (i32 j = 0; j < L2_SIZE; ++j)
                        for (i32 k = 0; k < L1_CHUNK_PER_32; ++k)
                            net.L1Weights[bucket][i * L1_CHUNK_PER_32 * L2_SIZE
                                                + j * L1_CHUNK_PER_32
                                                + k] = tempL1[i * L1_CHUNK_PER_32 + k][bucket][j];

                for (i32 i = 0; i < L2_SIZE; ++i)
                    net.L1Biases[bucket][i] = UQNet->L1Biases[bucket][i];

                for (i32 i = 0; i < L2_SIZE; ++i)
                    for (i32 j = 0; j < L3_SIZE; ++j)
                        net.L2Weights[bucket][i * L3_SIZE + j] = UQNet->L2Weights[i][bucket][j];

                for (i32 i = 0; i < L3_SIZE; ++i)
                    net.L2Biases[bucket][i] = UQNet->L2Biases[bucket][i];

                for (i32 i = 0; i < L3_SIZE; ++i)
                    net.L3Weights[bucket][i] = UQNet->L3Weights[i][bucket];

                net.L3Biases[bucket] = UQNet->L3Biases[bucket];
            }

            const i32 numRegi = 4;
            const i32 numChunks = (32 / 2) / sizeof(i16);
            constexpr i32 order[] = {0, 2, 1, 3};
            __m128i regi[numRegi] = {};
            auto ws = reinterpret_cast<__m128i*>(&net.FTWeights);
            auto bs = reinterpret_cast<__m128i*>(&net.FTBiases);

            for (i32 i = 0; i < INPUT_SIZE * L1_SIZE * INPUT_BUCKETS / numChunks; i += numRegi)
            {
                for (i32 j = 0; j < numRegi; j++)
                    regi[j] = ws[i + j];

                for (i32 j = 0; j < numRegi; j++)
                    ws[i + j] = regi[order[j]];
            }

            for (i32 i = 0; i < L1_SIZE / numChunks; i += numRegi)
            {
                for (i32 j = 0; j < numRegi; j++)
                    regi[j] = bs[i + j];

                for (i32 j = 0; j < numRegi; j++)
                    bs[i + j] = regi[order[j]];
            }

#if !defined(_MSC_VER)
            delete[] tempL1;
#endif

            g_network = &net;
        }

        static void SetupNNZ()
        {
            for (u32 i = 0; i < 256; i++)
            {
                u16* ptr = reinterpret_cast<u16*>(&nnzTable.Entries[i]);

                u32 j = i;
                u32 k = 0;
                while (j != 0)
                {
                    u32 lsbIndex = std::countr_zero(j);
                    j &= j - 1;
                    ptr[k++] = (u16)lsbIndex;
                }
            }
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

            accumulator.Sides[perspective] = g_network->FTBiases;
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
                const auto weights = &g_network->FTWeights[idx];
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

                        const auto weights = &g_network->FTWeights[idx];
                        Add(ourAccumulation, ourAccumulation, weights);
                    }

                    while (removed != 0)
                    {
                        i32 sq = poplsb(removed);
                        i32 idx = FeatureIndexSingle(pc, pt, sq, ourKing, perspective);

                        const auto weights = &g_network->FTWeights[idx];
                        Sub(ourAccumulation, ourAccumulation, weights);
                    }
                }
            }

            entryAcc.CopyTo(&accumulator, perspective);
            bb.CopyTo(entryBB);

            accumulator.Computed[perspective] = true;
        }


        i32 GetEvaluation(Position& pos) {
            const auto occ = popcount(pos.bb.Occupancy);
            const auto outputBucket = (occ - 2) / ((32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS);

            return GetEvaluation(pos, outputBucket);
        }

        i32 GetEvaluation(Position& pos, i32 outputBucket) {
            Bitboard& bb = pos.bb;
            Accumulator& accumulator = *pos.State->accumulator;
            ProcessUpdates(pos);

            float L1Outputs[L2_SIZE];
            float L2Outputs[L3_SIZE];
            float L3Output = 0;

            auto us = Span<i16>(accumulator.Sides[pos.ToMove]);
            auto them = Span<i16>(accumulator.Sides[Not(pos.ToMove)]);

            ActivateFTSparse(us, them, Span<i8>(net.L1Weights[outputBucket]), Span<float>(net.L1Biases[outputBucket]), L1Outputs);
            ActivateL2(L1Outputs, Span<float>(net.L2Weights[outputBucket]), Span<float>(net.L2Biases[outputBucket]), L2Outputs);
            ActivateL3(L2Outputs, Span<float>(net.L3Weights[outputBucket]), net.L3Biases[outputBucket], L3Output);

            return (i32)(L3Output * OutputScale);
        }



        static void ActivateFTSparse(Span<i16> us, Span<i16> them, Span<i8> weights, Span<float> biases, Span<float> output) {
            const auto ft_zero = _mm256_setzero_si256();
            const auto ft_one = _mm256_set1_epi16(FT_QUANT);

            i32 nnzCount = 0;
            i32 offset = 0;

            i8 ft_outputs[L1_SIZE];
            u16 nnzIndices[L1_SIZE / L1_CHUNK_PER_32];

            const vec_128i baseInc = _mm_set1_epi16((u16)8);
            vec_128i baseVec = _mm_setzero_si128();

            
            for (const auto acc : { us, them }) {
                for (i32 i = 0; i < L1_PAIR_COUNT; i += (I16_CHUNK_SIZE * 2)) {
                    const auto input0a = _mm256_load_si256(reinterpret_cast<const vec_i16*>(&acc[i + 0 * I16_CHUNK_SIZE + 0]));
                    const auto input0b = _mm256_load_si256(reinterpret_cast<const vec_i16*>(&acc[i + 1 * I16_CHUNK_SIZE + 0]));

                    const auto input1a = _mm256_load_si256(reinterpret_cast<const vec_i16*>(&acc[i + 0 * I16_CHUNK_SIZE + L1_PAIR_COUNT]));
                    const auto input1b = _mm256_load_si256(reinterpret_cast<const vec_i16*>(&acc[i + 1 * I16_CHUNK_SIZE + L1_PAIR_COUNT]));

                    const auto clipped0a = _mm256_min_epi16(_mm256_max_epi16(input0a, ft_zero), ft_one);
                    const auto clipped0b = _mm256_min_epi16(_mm256_max_epi16(input0b, ft_zero), ft_one);

                    const auto clipped1a = _mm256_min_epi16(input1a, ft_one);
                    const auto clipped1b = _mm256_min_epi16(input1b, ft_one);

                    const auto producta = _mm256_mulhi_epi16(_mm256_slli_epi16(clipped0a, 16 - FT_SHIFT), clipped1a);
                    const auto productb = _mm256_mulhi_epi16(_mm256_slli_epi16(clipped0b, 16 - FT_SHIFT), clipped1b);

                    const auto product_one = _mm256_packus_epi16(producta, productb);
                    _mm256_storeu_si256(reinterpret_cast<vec_i8*>(&ft_outputs[offset + i]), product_one);

                    const auto nnz_mask = vec_nnz_mask(product_one);

                    for (i32 j = 0; j < NNZ_OUTPUTS_PER_CHUNK; j++) {
                        i32 lookup = (nnz_mask >> (j * 8)) & 0xFF;
                        auto offsets = nnzTable.Entries[lookup];
                        _mm_storeu_si128(reinterpret_cast<vec_128i*>(&nnzIndices[nnzCount]), _mm_add_epi16(baseVec, offsets));

                        nnzCount += std::popcount((u32)lookup);
                        baseVec = _mm_add_epi16(baseVec, baseInc);
                    }

                }

                offset += L1_PAIR_COUNT;
            }

#if defined(PERM_COUNT)
            EvalCalls++;
            ActivationCount += static_cast<u64>(nnzCount);
            for (i32 i = 0; i < L1_SIZE; i++)
                NNZCounts[i] += (ft_outputs[i] ? 1UL : 0);
#endif

            ActivateL1Sparse(Span<i8>(ft_outputs), weights, biases, output, Span<u16>(nnzIndices), nnzCount);
        }


        static void ActivateL1Sparse(Span<i8> inputs, Span<i8> weights, Span<float> biases, Span<float> output, Span<u16> nnzIndices, const i32 nnzCount) {
            vec_i32 sums[L2_SIZE / I32_CHUNK_SIZE]{};

            const auto inputs32 = (i32*)(inputs.data());
            for (i32 i = 0; i < nnzCount; i++) {
                const auto index = nnzIndices[i];
                const auto input32 = _mm256_set1_epi32(inputs32[index]);
                const auto weight = reinterpret_cast<const vec_i8*>(&weights[index * L1_CHUNK_PER_32 * L2_SIZE]);
                for (i32 k = 0; k < L2_SIZE / F32_CHUNK_SIZE; k++)
                    sums[k] = vec_dpbusd_epi32(sums[k], input32, weight[k]);
            }

            const auto zero = _mm256_set1_ps(0.0f);
            const auto one = _mm256_set1_ps(1.0f);

            const auto sumMul = _mm256_set1_ps((1 << FT_SHIFT) / (float)(FT_QUANT * FT_QUANT * L1_QUANT));
            for (i32 i = 0; i < L2_SIZE / F32_CHUNK_SIZE; ++i) {
                const auto biasVec = _mm256_loadu_ps(&biases[i * F32_CHUNK_SIZE]);
                const auto sumPs = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums[i]), sumMul, biasVec);
                const auto clipped = _mm256_min_ps(_mm256_max_ps(sumPs, zero), one);
                const auto squared = _mm256_mul_ps(clipped, clipped);
                _mm256_storeu_ps(&output[i * F32_CHUNK_SIZE], squared);
            }
        }


        static void ActivateL2(Span<float> inputs, Span<float> weights, Span<float> biases, Span<float> output) {
            vec_ps32 sumVecs[L3_SIZE / F32_CHUNK_SIZE];

            for (i32 i = 0; i < L3_SIZE / F32_CHUNK_SIZE; ++i)
                sumVecs[i] = _mm256_loadu_ps(&biases[i * F32_CHUNK_SIZE]);

            for (i32 i = 0; i < L2_SIZE; ++i) {
                const auto inputVec = _mm256_set1_ps(inputs[i]);
                const auto weight = reinterpret_cast<const vec_ps32*>(&weights[i * L3_SIZE]);
                for (i32 j = 0; j < L3_SIZE / F32_CHUNK_SIZE; ++j)
                    sumVecs[j] = vec_mul_add_ps(inputVec, weight[j], sumVecs[j]);
            }

            const auto zero = _mm256_set1_ps(0.0f);
            const auto one = _mm256_set1_ps(1.0f);
            for (i32 i = 0; i < L3_SIZE / F32_CHUNK_SIZE; ++i) {
                const auto clipped = _mm256_min_ps(_mm256_max_ps(sumVecs[i], zero), one);
                const auto squared = _mm256_mul_ps(clipped, clipped);
                _mm256_storeu_ps(&output[i * F32_CHUNK_SIZE], squared);
            }
        }


        static void ActivateL3(Span<float> inputs, Span<float> weights, const float bias, float& output) {
            auto sumVec = _mm256_set1_ps(0.0f);

            for (i32 i = 0; i < L3_SIZE / F32_CHUNK_SIZE; i++) {
                const auto weightVec = _mm256_loadu_ps(&weights[i * F32_CHUNK_SIZE]);
                const auto inputsVec = _mm256_loadu_ps(&inputs[i * F32_CHUNK_SIZE]);
                sumVec = vec_mul_add_ps(inputsVec, weightVec, sumVec);
            }

            output = bias + scuffed_chatgpt_hsum(sumVec);
        }


        void MakeMoveNN(Position& pos, Move m) {
            Bitboard& bb = pos.bb;

            Accumulator* src = pos.State->accumulator;
            Accumulator* dst = pos.NextState()->accumulator;

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
            auto FeatureWeights = reinterpret_cast<const i16*>(&g_network->FTWeights[0]);
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

            return { whiteIndex * L1_SIZE, blackIndex * L1_SIZE };
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

            return ((768 * KingBuckets[kingSq]) + ((pc ^ perspective) * ColorStride) + (pt * PieceStride) + (sq)) * L1_SIZE;
        }


        void ResetCaches(Position& pos) {
            for (auto& bucket : pos.CachedBuckets) {
                bucket.accumulator.Sides[WHITE] = bucket.accumulator.Sides[BLACK] = g_network->FTBiases;
                bucket.Boards[WHITE].Reset();
                bucket.Boards[BLACK].Reset();
            }
        }


        static void PermuteFT(Span<i16> ftWeights, Span<i16> ftBiases)
        {
            const i32 OneBucket = (INPUT_SIZE * L1_SIZE);
            std::vector<i16> temp(OneBucket, 0);

            for (i32 bucket = 0; bucket < INPUT_BUCKETS; bucket++)
            {
                Span<i16> ftBucket = Span<i16>(&ftWeights[(bucket * OneBucket)], OneBucket);

                for (size_t i = 0; i < OneBucket; i++)
                    temp[i] = ftBucket[i];

                for (i32 i = 0; i < INPUT_SIZE; i++)
                {
                    for (i32 dst = 0; dst < L1_PAIR_COUNT; dst++)
                    {
                        i32 src = PermuteIndices[dst];
                        auto f = i * L1_SIZE;

                        ftBucket[f + dst] = temp[f + src];
                        ftBucket[f + dst + L1_PAIR_COUNT] = temp[f + src + L1_PAIR_COUNT];
                    }
                }
            }

            std::vector<i16> temp2(L1_SIZE, 0);
            for (i32 i = 0; i < L1_SIZE; i++)
                temp2[i] = ftBiases[i];

            for (i32 dst = 0; dst < L1_PAIR_COUNT; dst++)
            {
                i32 src = PermuteIndices[dst];

                ftBiases[dst] = temp2[src];
                ftBiases[dst + L1_PAIR_COUNT] = temp2[src + L1_PAIR_COUNT];
            }
        }


        static void PermuteL1(i8 l1Weights[L1_SIZE][OUTPUT_BUCKETS][L2_SIZE])
        {
            auto temp = new i8[L1_SIZE][OUTPUT_BUCKETS][L2_SIZE];
            std::memcpy(temp, l1Weights, N_L1W);

            for (i32 dst = 0; dst < L1_PAIR_COUNT; dst++)
            {
                i32 src = PermuteIndices[dst];

                for (i32 b = 0; b < OUTPUT_BUCKETS; b++)
                {
                    for (i32 l2 = 0; l2 < L2_SIZE; l2++)
                    {
                        l1Weights[dst][b][l2] = temp[src][b][l2];
                        l1Weights[dst + L1_PAIR_COUNT][b][l2] = temp[src + L1_PAIR_COUNT][b][l2];
                    }
                }
            }

            delete[] temp;
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
