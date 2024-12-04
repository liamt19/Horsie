
#include "../nnue/nn.h"

#include <span>
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
                
            sbyte* l1Ptr = reinterpret_cast<sbyte*>(UQNet->L1Weights);
            for (size_t i = 0; i < L1_SIZE * L2_SIZE * OUTPUT_BUCKETS; i++)
                l1Ptr[i] = read_little_endian<sbyte>(stream);

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
            auto tempL1 = new sbyte[L1_SIZE][OUTPUT_BUCKETS][L2_SIZE];
            std::memcpy(tempL1, UQNet->L1Weights, L1_SIZE * OUTPUT_BUCKETS * L2_SIZE);
#endif

            for (size_t i = 0; i < INPUT_SIZE * L1_SIZE * INPUT_BUCKETS; i++)
                net.FTWeights[i] = UQNet->FTWeights[i];

            for (size_t i = 0; i < L1_SIZE; i++)
                net.FTBiases[i] = UQNet->FTBiases[i];

            PermuteFT(net.FTWeights, net.FTBiases);
            PermuteL1(tempL1);

            for (int bucket = 0; bucket < OUTPUT_BUCKETS; bucket++)
            {
                for (int i = 0; i < L1_SIZE / L1_CHUNK_PER_32; ++i)
                    for (int j = 0; j < L2_SIZE; ++j)
                        for (int k = 0; k < L1_CHUNK_PER_32; ++k)
                            net.L1Weights[bucket][i * L1_CHUNK_PER_32 * L2_SIZE
                                                + j * L1_CHUNK_PER_32
                                                + k] = tempL1[i * L1_CHUNK_PER_32 + k][bucket][j];

                for (int i = 0; i < L2_SIZE; ++i)
                    net.L1Biases[bucket][i] = UQNet->L1Biases[bucket][i];

                for (int i = 0; i < L2_SIZE; ++i)
                    for (int j = 0; j < L3_SIZE; ++j)
                        net.L2Weights[bucket][i * L3_SIZE + j] = UQNet->L2Weights[i][bucket][j];

                for (int i = 0; i < L3_SIZE; ++i)
                    net.L2Biases[bucket][i] = UQNet->L2Biases[bucket][i];

                for (int i = 0; i < L3_SIZE; ++i)
                    net.L3Weights[bucket][i] = UQNet->L3Weights[i][bucket];

                net.L3Biases[bucket] = UQNet->L3Biases[bucket];
            }

            const int numRegi = 4;
            const int numChunks = (32 / 2) / sizeof(short);
            constexpr int order[] = {0, 2, 1, 3};
            __m128i regi[numRegi] = {};
            auto ws = reinterpret_cast<__m128i*>(&net.FTWeights);
            auto bs = reinterpret_cast<__m128i*>(&net.FTBiases);

            for (int i = 0; i < INPUT_SIZE * L1_SIZE * INPUT_BUCKETS / numChunks; i += numRegi)
            {
                for (int j = 0; j < numRegi; j++)
                    regi[j] = ws[i + j];

                for (int j = 0; j < numRegi; j++)
                    ws[i + j] = regi[order[j]];
            }

            for (int i = 0; i < L1_SIZE / numChunks; i += numRegi)
            {
                for (int j = 0; j < numRegi; j++)
                    regi[j] = bs[i + j];

                for (int j = 0; j < numRegi; j++)
                    bs[i + j] = regi[order[j]];
            }

#if !defined(_MSC_VER)
            delete[] tempL1;
#endif

            g_network = &net;
        }

        static void SetupNNZ()
        {
            for (uint i = 0; i < 256; i++)
            {
                ushort* ptr = reinterpret_cast<ushort*>(&nnzTable.Entries[i]);

                uint j = i;
                uint k = 0;
                while (j != 0)
                {
                    uint lsbIndex = std::countr_zero(j);
                    j &= j - 1;
                    ptr[k++] = (ushort)lsbIndex;
                }
            }
        }


        void RefreshAccumulator(Position& pos)
        {
            RefreshAccumulatorPerspectiveFull(pos, WHITE);
            RefreshAccumulatorPerspectiveFull(pos, BLACK);
        }


        void RefreshAccumulatorPerspectiveFull(Position& pos, int perspective)
        {
            Accumulator& accumulator = *pos.State->accumulator;
            Bitboard& bb = pos.bb;

            accumulator.Sides[perspective] = g_network->FTBiases;
            accumulator.NeedsRefresh[perspective] = false;
            accumulator.Computed[perspective] = true;

            int ourKing = pos.State->KingSquares[perspective];
            ulong occ = bb.Occupancy;
            while (occ != 0)
            {
                int pieceIdx = poplsb(occ);

                int pt = bb.GetPieceAtIndex(pieceIdx);
                int pc = bb.GetColorAtIndex(pieceIdx);

                int idx = FeatureIndexSingle(pc, pt, pieceIdx, ourKing, perspective);

                const auto accum   = reinterpret_cast<short*>(&accumulator.Sides[perspective]);
                const auto weights = &g_network->FTWeights[idx];
                Add(accum, accum, weights);
            }

            BucketCache& cache = pos.CachedBuckets[BucketForPerspective(ourKing, perspective)];
            Bitboard& entryBB = cache.Boards[perspective];
            Accumulator& entryAcc = cache.accumulator;

            accumulator.CopyTo(&entryAcc, perspective);
            bb.CopyTo(entryBB);
        }


        void RefreshAccumulatorPerspective(Position& pos, int perspective)
        {
            Accumulator& accumulator = *pos.State->accumulator;
            Bitboard& bb = pos.bb;

            int ourKing = pos.State->KingSquares[perspective];
            int thisBucket = KingBuckets[ourKing];

            BucketCache& rtEntry = pos.CachedBuckets[BucketForPerspective(ourKing, perspective)];
            Bitboard& entryBB = rtEntry.Boards[perspective];
            Accumulator& entryAcc = rtEntry.accumulator;

            auto ourAccumulation = reinterpret_cast<short*>(&entryAcc.Sides[perspective]);
            accumulator.NeedsRefresh[perspective] = false;

            for (int pc = 0; pc < COLOR_NB; pc++)
            {
                for (int pt = 0; pt < PIECE_NB; pt++)
                {
                    ulong prev = entryBB.Pieces[pt] & entryBB.Colors[pc];
                    ulong curr =      bb.Pieces[pt] &      bb.Colors[pc];

                    ulong added   = curr & ~prev;
                    ulong removed = prev & ~curr;

                    while (added != 0)
                    {
                        int sq = poplsb(added);
                        int idx = FeatureIndexSingle(pc, pt, sq, ourKing, perspective);

                        const auto weights = &g_network->FTWeights[idx];
                        Add(ourAccumulation, ourAccumulation, weights);
                    }

                    while (removed != 0)
                    {
                        int sq = poplsb(removed);
                        int idx = FeatureIndexSingle(pc, pt, sq, ourKing, perspective);

                        const auto weights = &g_network->FTWeights[idx];
                        Sub(ourAccumulation, ourAccumulation, weights);
                    }
                }
            }

            entryAcc.CopyTo(&accumulator, perspective);
            bb.CopyTo(entryBB);

            accumulator.Computed[perspective] = true;
        }


        int GetEvaluation(Position& pos)
        {
            int occ = (int)popcount(pos.bb.Occupancy);
            int outputBucket = (occ - 2) / ((32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS);

            return GetEvaluation(pos, outputBucket);
        }

        int GetEvaluation(Position& pos, int outputBucket)
        {
            Bitboard& bb = pos.bb;
            Accumulator& accumulator = *pos.State->accumulator;
            ProcessUpdates(pos);

            float L1Outputs[L2_SIZE];
            float L2Outputs[L3_SIZE];
            float L3Output = 0;

            auto us = reinterpret_cast<short*>(&accumulator.Sides[pos.ToMove]);
            auto them = reinterpret_cast<short*>(&accumulator.Sides[Not(pos.ToMove)]);

            ActivateFTSparse(us, them, reinterpret_cast<sbyte*>(&net.L1Weights[outputBucket]), reinterpret_cast<float*>(&net.L1Biases[outputBucket]), L1Outputs);
            ActivateL2(L1Outputs, reinterpret_cast<float*>(&net.L2Weights[outputBucket]), reinterpret_cast<float*>(&net.L2Biases[outputBucket]), L2Outputs);
            ActivateL3(L2Outputs, reinterpret_cast<float*>(&net.L3Weights[outputBucket]), net.L3Biases[outputBucket], L3Output);

            return (int)(L3Output * OutputScale);
        }



        static void ActivateFTSparse(short* us, short* them, sbyte* weights, float* biases, float* output)
        {
            const auto ft_zero = _mm256_setzero_si256();
            const auto ft_one = _mm256_set1_epi16(FT_QUANT);

            int nnzCount = 0;
            int offset = 0;

            sbyte ft_outputs[L1_SIZE];
            ushort nnzIndices[L1_SIZE / L1_CHUNK_PER_32];

            const vec_128i baseInc = _mm_set1_epi16((ushort)8);
            vec_128i baseVec = _mm_setzero_si128();

            for (int perspective = 0; perspective < 2; perspective++)
            {
                short* acc = perspective == 0 ? us : them;

                for (int i = 0; i < L1_PAIR_COUNT; i += (I16_CHUNK_SIZE * 2))
                {
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

                    for (int j = 0; j < NNZ_OUTPUTS_PER_CHUNK; j++)
                    {
                        int lookup = (nnz_mask >> (j * 8)) & 0xFF;
                        auto offsets = nnzTable.Entries[lookup];
                        _mm_storeu_si128(reinterpret_cast<vec_128i*>(&nnzIndices[nnzCount]), _mm_add_epi16(baseVec, offsets));

                        nnzCount += std::popcount((uint)lookup);
                        baseVec = _mm_add_epi16(baseVec, baseInc);
                    }

                }

                offset += L1_PAIR_COUNT;
            }

#if defined(PERM_COUNT)
            EvalCalls++;
            ActivationCount += (ulong)nnzCount;
            lock(NNZCounts)
            {
                for (int i = 0; i < L1_SIZE; i++)
                    NNZCounts[i] += (ft_outputs[i] != 0) ? 1UL : 0;
            }
#endif

            ActivateL1Sparse(ft_outputs, weights, biases, output, nnzIndices, nnzCount);
        }


        static void ActivateL1Sparse(sbyte* inputs, sbyte* weights, float* biases, float* output, ushort* nnzIndices, int nnzCount)
        {
            vec_i32 sums[L2_SIZE / I32_CHUNK_SIZE]{};

            const int* inputs32 = (int*)(inputs);
            for (int i = 0; i < nnzCount; i++)
            {
                const auto index = nnzIndices[i];
                const auto input32 = _mm256_set1_epi32(inputs32[index]);
                const auto weight = reinterpret_cast<const vec_i8*>(&weights[index * L1_CHUNK_PER_32 * L2_SIZE]);
                for (int k = 0; k < L2_SIZE / F32_CHUNK_SIZE; k++)
                {
                    sums[k] = vec_dpbusd_epi32(sums[k], input32, weight[k]);
                }
            }

            const auto zero = _mm256_set1_ps(0.0f);
            const auto one = _mm256_set1_ps(1.0f);

            const auto sumMul = _mm256_set1_ps((1 << FT_SHIFT) / (float)(FT_QUANT * FT_QUANT * L1_QUANT));
            for (int i = 0; i < L2_SIZE / F32_CHUNK_SIZE; ++i)
            {
                const auto biasVec = _mm256_loadu_ps(&biases[i * F32_CHUNK_SIZE]);
                const auto sumPs = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sums[i]), sumMul, biasVec);
                const auto clipped = _mm256_min_ps(_mm256_max_ps(sumPs, zero), one);
                const auto squared = _mm256_mul_ps(clipped, clipped);
                _mm256_storeu_ps(&output[i * F32_CHUNK_SIZE], squared);
            }
        }


        static void ActivateL2(float* inputs, float* weights, float* biases, float* output)
        {
            vec_ps32 sumVecs[L3_SIZE / F32_CHUNK_SIZE];

            for (int i = 0; i < L3_SIZE / F32_CHUNK_SIZE; ++i)
                sumVecs[i] = _mm256_loadu_ps(&biases[i * F32_CHUNK_SIZE]);

            for (int i = 0; i < L2_SIZE; ++i)
            {
                const auto inputVec = _mm256_set1_ps(inputs[i]);
                const auto weight = reinterpret_cast<const vec_ps32*>(&weights[i * L3_SIZE]);
                for (int j = 0; j < L3_SIZE / F32_CHUNK_SIZE; ++j)
                {
                    sumVecs[j] = vec_mul_add_ps(inputVec, weight[j], sumVecs[j]);
                }
            }

            const auto zero = _mm256_set1_ps(0.0f);
            const auto one = _mm256_set1_ps(1.0f);
            for (int i = 0; i < L3_SIZE / F32_CHUNK_SIZE; ++i)
            {
                const auto clipped = _mm256_min_ps(_mm256_max_ps(sumVecs[i], zero), one);
                const auto squared = _mm256_mul_ps(clipped, clipped);
                _mm256_storeu_ps(&output[i * F32_CHUNK_SIZE], squared);
            }
        }


        static void ActivateL3(float* inputs, float* weights, float bias, float& output)
        {
            auto sumVec = _mm256_set1_ps(0.0f);

            for (int i = 0; i < L3_SIZE / F32_CHUNK_SIZE; i++)
            {
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

            int moveTo = m.To();
            int moveFrom = m.From();

            int us = pos.ToMove;
            int ourPiece = bb.GetPieceAtIndex(moveFrom);

            int them = Not(us);
            int theirPiece = bb.GetPieceAtIndex(moveTo);

            PerspectiveUpdate& wUpdate = dst->Update[WHITE];
            PerspectiveUpdate& bUpdate = dst->Update[BLACK];

            wUpdate.Clear();
            bUpdate.Clear();

            if (ourPiece == KING && (KingBuckets[moveFrom ^ (56 * us)] != KingBuckets[moveTo ^ (56 * us)]))
            {
                //  We will need to fully refresh our perspective, but we can still do theirs.
                dst->NeedsRefresh[us] = true;

                PerspectiveUpdate& theirUpdate = dst->Update[them];
                int theirKing = pos.State->KingSquares[them];

                int from = FeatureIndexSingle(us, ourPiece, moveFrom, theirKing, them);
                int to = FeatureIndexSingle(us, ourPiece, moveTo, theirKing, them);

                if (theirPiece != NONE && !m.IsCastle())
                {
                    int cap = FeatureIndexSingle(them, theirPiece, moveTo, theirKing, them);

                    theirUpdate.PushSubSubAdd(from, cap, to);
                }
                else if (m.IsCastle())
                {
                    int rookFromSq = moveTo;
                    int rookToSq = m.CastlingRookSquare();

                    to = FeatureIndexSingle(us, ourPiece, m.CastlingKingSquare(), theirKing, them);

                    int rookFrom = FeatureIndexSingle(us, ROOK, rookFromSq, theirKing, them);
                    int rookTo = FeatureIndexSingle(us, ROOK, rookToSq, theirKing, them);

                    theirUpdate.PushSubSubAddAdd(from, rookFrom, to, rookTo);
                }
                else
                {
                    theirUpdate.PushSubAdd(from, to);
                }
            }
            else
            {
                int wKing = pos.State->KingSquares[WHITE];
                int bKing = pos.State->KingSquares[BLACK];

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
                    int idxPawn = moveTo - ShiftUpDir(us);

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
            for (int perspective = 0; perspective < 2; perspective++)
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


        void UpdateSingle(Accumulator* prev, Accumulator* curr, int perspective)
        {
            auto FeatureWeights = reinterpret_cast<const short*>(&g_network->FTWeights[0]);
            const auto& updates = curr->Update[perspective];

            if (updates.AddCnt == 0 && updates.SubCnt == 0)
            {
                //  For null moves, we still need to carry forward the correct accumulator state
                prev->CopyTo(curr, perspective);
                return;
            }

            auto src = reinterpret_cast<short*>(&prev->Sides[perspective]);
            auto dst = reinterpret_cast<short*>(&curr->Sides[perspective]);
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


        std::pair<int, int> FeatureIndex(int pc, int pt, int sq, int wk, int bk)
        {
            const int ColorStride = 64 * 6;
            const int PieceStride = 64;

            int wSq = sq;
            int bSq = sq ^ 56;

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

            int whiteIndex = (768 * KingBuckets[wk]) + (pc * ColorStride) + (pt * PieceStride) + wSq;
            int blackIndex = (768 * KingBuckets[bk]) + (Not(pc) * ColorStride) + (pt * PieceStride) + bSq;

            return { whiteIndex * L1_SIZE, blackIndex * L1_SIZE };
        }


        int FeatureIndexSingle(int pc, int pt, int sq, int kingSq, int perspective)
        {
            const int ColorStride = 64 * 6;
            const int PieceStride = 64;

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


        static void PermuteFT(std::array<short, INPUT_SIZE * L1_SIZE * INPUT_BUCKETS>& ftWeights, std::array<short, L1_SIZE>& ftBiases)
        {
            const int OneBucket = (INPUT_SIZE * L1_SIZE);
            std::vector<short> temp(OneBucket, 0);

            for (int bucket = 0; bucket < INPUT_BUCKETS; bucket++)
            {
                std::span<short> ftBucket = std::span<short>(&ftWeights[(bucket * OneBucket)], OneBucket);

                for (size_t i = 0; i < OneBucket; i++)
                    temp[i] = ftBucket[i];

                for (int i = 0; i < INPUT_SIZE; i++)
                {
                    for (int dst = 0; dst < L1_PAIR_COUNT; dst++)
                    {
                        int src = PermuteIndices[dst];
                        auto f = i * L1_SIZE;

                        ftBucket[f + dst] = temp[f + src];
                        ftBucket[f + dst + L1_PAIR_COUNT] = temp[f + src + L1_PAIR_COUNT];
                    }
                }
            }

            std::vector<short> temp2(L1_SIZE, 0);
            for (int i = 0; i < L1_SIZE; i++)
                temp2[i] = ftBiases[i];

            for (int dst = 0; dst < L1_PAIR_COUNT; dst++)
            {
                int src = PermuteIndices[dst];

                ftBiases[dst] = temp2[src];
                ftBiases[dst + L1_PAIR_COUNT] = temp2[src + L1_PAIR_COUNT];
            }
        }


        static void PermuteL1(sbyte l1Weights[L1_SIZE][OUTPUT_BUCKETS][L2_SIZE])
        {
            auto temp = new sbyte[L1_SIZE][OUTPUT_BUCKETS][L2_SIZE];
            std::memcpy(temp, l1Weights, N_L1W);

            for (int dst = 0; dst < L1_PAIR_COUNT; dst++)
            {
                int src = PermuteIndices[dst];

                for (int b = 0; b < OUTPUT_BUCKETS; b++)
                {
                    for (int l2 = 0; l2 < L2_SIZE; l2++)
                    {
                        l1Weights[dst][b][l2] = temp[src][b][l2];
                        l1Weights[dst + L1_PAIR_COUNT][b][l2] = temp[src + L1_PAIR_COUNT][b][l2];
                    }
                }
            }

            delete[] temp;
        }


        static void SubSubAddAdd(const short* _src, short* _dst, const short* _sub1, const short* _sub2, const short* _add1, const short* _add2) {
            const __m256i* src  = reinterpret_cast<const __m256i*>(_src);
                  __m256i* dst  = reinterpret_cast<      __m256i*>(_dst);
            const __m256i* sub1 = reinterpret_cast<const __m256i*>(_sub1);
            const __m256i* sub2 = reinterpret_cast<const __m256i*>(_sub2);
            const __m256i* add1 = reinterpret_cast<const __m256i*>(_add1);
            const __m256i* add2 = reinterpret_cast<const __m256i*>(_add2);
            for (int i = 0; i < SIMD_CHUNKS; i++)
            {
                dst[i] = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(src[i], add1[i]), add2[i]), sub1[i]), sub2[i]);
            }
        }

        static void SubSubAdd(const short* _src, short* _dst, const short* _sub1, const short* _sub2, const short* _add1) {
            const __m256i* src  = reinterpret_cast<const __m256i*>(_src);
                  __m256i* dst  = reinterpret_cast<      __m256i*>(_dst);
            const __m256i* sub1 = reinterpret_cast<const __m256i*>(_sub1);
            const __m256i* sub2 = reinterpret_cast<const __m256i*>(_sub2);
            const __m256i* add1 = reinterpret_cast<const __m256i*>(_add1);
            for (int i = 0; i < SIMD_CHUNKS; i++)
            {
                dst[i] = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(src[i], add1[i]), sub1[i]), sub2[i]);
            }
        }

        static void SubAdd(const short* _src, short* _dst, const short* _sub1, const short* _add1) {
            const __m256i* src  = reinterpret_cast<const __m256i*>(_src);
                  __m256i* dst  = reinterpret_cast<      __m256i*>(_dst);
            const __m256i* sub1 = reinterpret_cast<const __m256i*>(_sub1);
            const __m256i* add1 = reinterpret_cast<const __m256i*>(_add1);
            for (int i = 0; i < SIMD_CHUNKS; i++)
            {
                dst[i] = _mm256_sub_epi16(_mm256_add_epi16(src[i], add1[i]), sub1[i]);
            }
        }

        static void Add(const short* _src, short* _dst, const short* _add1) {
            const __m256i* src  = reinterpret_cast<const __m256i*>(_src);
                  __m256i* dst  = reinterpret_cast<      __m256i*>(_dst);
            const __m256i* add1 = reinterpret_cast<const __m256i*>(_add1);
            for (int i = 0; i < SIMD_CHUNKS; i++)
                dst[i] = _mm256_add_epi16(src[i], add1[i]);
        }

        static void Sub(const short* _src, short* _dst, const short* _sub1) {
            const __m256i* src  = reinterpret_cast<const __m256i*>(_src);
                  __m256i* dst  = reinterpret_cast<      __m256i*>(_dst);
            const __m256i* sub1 = reinterpret_cast<const __m256i*>(_sub1);
            for (int i = 0; i < SIMD_CHUNKS; i++)
                dst[i] = _mm256_sub_epi16(src[i], sub1[i]);
        }
    }

}
