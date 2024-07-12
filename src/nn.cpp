

#include "nn.h"
#include "position.h"
#include "immintrin.h"

#include <fstream>
#include <cstring>

#include "./incbin/incbin.h"

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

        static int32_t hsum_8x32(__m256i v);
        static void SubAdd(__m256i* src, const __m256i* sub1, const __m256i* add1);
        static void SubSubAdd(__m256i* src, const __m256i* sub1, const __m256i* sub2, const __m256i* add1);
        static void SubSubAddAdd(__m256i* src, const __m256i* sub1, const __m256i* sub2, const __m256i* add1, const __m256i* add2);

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

#endif

            //auto temp = net.LayerWeights;
            //for (int bucket = 0; bucket < OutputBuckets; bucket++)
            //{
            //    for (int i = 0; i < HiddenSize * 2; i++)
            //    {
            //        net.LayerWeights[bucket][i] = temp[i][bucket];
            //    }
            //}
        }


        static constexpr int BucketForPerspective(int ksq, int perspective) {
            return (KingBuckets[(ksq ^ (56 * perspective))]);
        }


        void RefreshAccumulator(Position pos)
        {
            RefreshAccumulatorPerspectiveFull(pos, WHITE);
            RefreshAccumulatorPerspectiveFull(pos, BLACK);
        }


        void RefreshAccumulatorPerspectiveFull(Position& pos, int perspective)
        {
            Accumulator& accumulator = *pos.State->accumulator;
            Bitboard& bb = pos.bb;

            accumulator.Sides[perspective] = net.FeatureBiases;
            accumulator.NeedsRefresh[perspective] = false;

            int ourKing = pos.State->KingSquares[perspective];
            ulong occ = bb.Occupancy;
            while (occ != 0)
            {
                int pieceIdx = poplsb(occ);

                int pt = bb.GetPieceAtIndex(pieceIdx);
                int pc = bb.GetColorAtIndex(pieceIdx);

                int idx = FeatureIndexSingle(pc, pt, pieceIdx, ourKing, perspective);

                const auto accum   = reinterpret_cast<__m256i*>(&accumulator.Sides[perspective]);
                const auto weights = reinterpret_cast<const __m256i*>(&net.FeatureWeights[idx]);
                Add(accum, accum, weights);
            }

            BucketCache& cache = pos.CachedBuckets[BucketForPerspective(ourKing, perspective)];
            Bitboard& entryBB = cache.Boards[perspective];
            Accumulator& entryAcc = *cache.accumulator;

            accumulator.CopyTo(&entryAcc, perspective);
            bb.CopyTo(entryBB);
        }


        void RefreshAccumulatorPerspective(Position& pos, int perspective)
        {
            Accumulator& accumulator = *pos.State->accumulator;
            Bitboard& bb = pos.bb;

            int ourKing = pos.State->KingSquares[perspective];
            int thisBucket = KingBuckets[ourKing];

            BucketCache rtEntry = pos.CachedBuckets[BucketForPerspective(ourKing, perspective)];
            Bitboard& entryBB = rtEntry.Boards[perspective];
            Accumulator& entryAcc = *rtEntry.accumulator;

            auto ourAccumulation = entryAcc[perspective];
            accumulator.NeedsRefresh[perspective] = false;

            for (int pc = 0; pc < COLOR_NB; pc++)
            {
                for (int pt = 0; pt < PIECE_NB; pt++)
                {
                    ulong prev = entryBB.Pieces[pt] & entryBB.Colors[pc];
                    ulong curr = bb.Pieces[pt] & bb.Colors[pc];

                    ulong added   = curr & ~prev;
                    ulong removed = prev & ~curr;

                    while (added != 0)
                    {
                        int sq = poplsb(added);
                        int idx = FeatureIndexSingle(pc, pt, sq, ourKing, perspective);

                        const auto accum = reinterpret_cast<__m256i*>(&accumulator.Sides[perspective]);
                        const auto weights = reinterpret_cast<const __m256i*>(&net.FeatureWeights[idx]);
                        Add(accum, accum, weights);
                    }

                    while (removed != 0)
                    {
                        int sq = poplsb(removed);
                        int idx = FeatureIndexSingle(pc, pt, sq, ourKing, perspective);

                        const auto accum = reinterpret_cast<__m256i*>(&accumulator.Sides[perspective]);
                        const auto weights = reinterpret_cast<const __m256i*>(&net.FeatureWeights[idx]);
                        Sub(accum, accum, weights);
                    }
                }
            }

            entryAcc.CopyTo(&accumulator, perspective);
            bb.CopyTo(entryBB);
        }



        void MakeMoveNN(Position& pos, Move m) {
            Bitboard bb = pos.bb;

            Accumulator* src = pos.State->accumulator;
            Accumulator* dst = pos.NextState()->accumulator;

            dst->NeedsRefresh = src->NeedsRefresh;

            int moveTo = m.To();
            int moveFrom = m.From();

            int us = pos.ToMove;
            int ourPiece = bb.GetPieceAtIndex(moveFrom);

            int them = Not(us);
            int theirPiece = bb.GetPieceAtIndex(moveTo);

            const auto srcWhite = reinterpret_cast<const __m256i*>(&src->Sides[WHITE]);
            const auto srcBlack = reinterpret_cast<const __m256i*>(&src->Sides[BLACK]);

            auto dstWhite = reinterpret_cast<__m256i*>(&dst->Sides[WHITE]);
            auto dstBlack = reinterpret_cast<__m256i*>(&dst->Sides[BLACK]);

            auto FeatureWeights = reinterpret_cast<const __m256i*>(&net.FeatureWeights[0]);

            if (ourPiece == KING && (KingBuckets[moveFrom ^ (56 * us)] != KingBuckets[moveTo ^ (56 * us)]))
            {
                //  We will need to fully refresh our perspective, but we can still do theirs.
                dst->NeedsRefresh[us] = true;

                const auto theirSrc = reinterpret_cast<const __m256i*>(&src->Sides[them]);
                const auto theirDst = reinterpret_cast<__m256i*>(&dst->Sides[them]);
                int theirKing = pos.State->KingSquares[them];

                int from = FeatureIndexSingle(us, ourPiece, moveFrom, theirKing, them);
                int to = FeatureIndexSingle(us, ourPiece, moveTo, theirKing, them);

                if (theirPiece != NONE && !m.IsCastle())
                {
                    int cap = FeatureIndexSingle(them, theirPiece, moveTo, theirKing, them);

                    SubSubAdd(theirSrc, theirDst,
                        (FeatureWeights + from),
                        (FeatureWeights + cap),
                        (FeatureWeights + to));
                }
                else if (m.IsCastle())
                {
                    int rookFromSq = moveTo;
                    int rookToSq = m.CastlingRookSquare();

                    to = FeatureIndexSingle(us, ourPiece, m.CastlingKingSquare(), theirKing, them);

                    int rookFrom = FeatureIndexSingle(us, Rook, rookFromSq, theirKing, them);
                    int rookTo = FeatureIndexSingle(us, Rook, rookToSq, theirKing, them);

                    SubSubAddAdd(theirSrc, theirDst,
                        (FeatureWeights + from),
                        (FeatureWeights + rookFrom),
                        (FeatureWeights + to),
                        (FeatureWeights + rookTo));
                }
                else
                {
                    SubAdd(theirSrc, theirDst,
                        (FeatureWeights + from),
                        (FeatureWeights + to));
                }
            }
            else
            {
                int wKing = pos.State->KingSquares[WHITE];
                int bKing = pos.State->KingSquares[BLACK];

                const auto [wFrom, bFrom] = FeatureIndex(us, ourPiece, moveFrom, wKing, bKing);
                const auto [wTo, bTo] = FeatureIndex(us, m.IsPromotion() ? m.PromotionTo() : ourPiece, moveTo, wKing, bKing);

                if (theirPiece != NONE)
                {
                    const auto [wCap, bCap] = FeatureIndex(them, theirPiece, moveTo, wKing, bKing);

                    SubSubAdd(srcWhite, dstWhite,
                        (FeatureWeights + wFrom),
                        (FeatureWeights + wCap),
                        (FeatureWeights + wTo));

                    SubSubAdd(srcBlack, dstBlack,
                        (FeatureWeights + bFrom),
                        (FeatureWeights + bCap),
                        (FeatureWeights + bTo));
                }
                else if (m.IsEnPassant())
                {
                    int idxPawn = moveTo - ShiftUpDir(us);

                    const auto [wCap, bCap] = FeatureIndex(them, Pawn, idxPawn, wKing, bKing);

                    SubSubAdd(srcWhite, dstWhite,
                        (FeatureWeights + wFrom),
                        (FeatureWeights + wCap),
                        (FeatureWeights + wTo));

                    SubSubAdd(srcBlack, dstBlack,
                        (FeatureWeights + bFrom),
                        (FeatureWeights + bCap),
                        (FeatureWeights + bTo));
                }
                else
                {
                    SubAdd(srcWhite, dstWhite,
                        (FeatureWeights + wFrom),
                        (FeatureWeights + wTo));

                    SubAdd(srcBlack, dstBlack,
                        (FeatureWeights + bFrom),
                        (FeatureWeights + bTo));
                }
            }
        }



        int GetEvaluation(Position& pos) {

            Accumulator& accumulator = *pos.State->accumulator;

            if (accumulator.NeedsRefresh[WHITE])
            {
                RefreshAccumulatorPerspective(pos, WHITE);
            }

            if (accumulator.NeedsRefresh[BLACK])
            {
                RefreshAccumulatorPerspective(pos, BLACK);
            }

            Bitboard& bb = pos.bb;
            
            const __m256i ClampMin = _mm256_setzero_si256();
            const __m256i ClampMax = _mm256_set1_epi16(QA);
            __m256i sum = _mm256_setzero_si256();

            int occ = (int)popcount(pos.bb.Occupancy);
            int outputBucket = std::min((63 - occ) * (32 - occ) / 225, 7);

            auto  stm = reinterpret_cast<const __m256i*>(&accumulator.Sides[pos.ToMove]);
            auto nstm = reinterpret_cast<const __m256i*>(&accumulator.Sides[Not(pos.ToMove)]);
            auto ourWeights   = reinterpret_cast<const __m256i*>(&net.LayerWeights[outputBucket]);
            auto theirWeights = reinterpret_cast<const __m256i*>(&net.LayerWeights[outputBucket][SIMD_CHUNKS]);

            for (int i = 0; i < SIMD_CHUNKS; i++) {
                __m256i clamp = _mm256_min_epi16(ClampMax, _mm256_max_epi16(ClampMin, stm[i]));
                __m256i mult = _mm256_mullo_epi16(clamp, ourWeights[i]);
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(mult, clamp));
            }

            for (int i = 0; i < SIMD_CHUNKS; i++) {
                __m256i clamp = _mm256_min_epi16(ClampMax, _mm256_max_epi16(ClampMin, nstm[i]));
                __m256i mult = _mm256_mullo_epi16(clamp, theirWeights[i]);
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(mult, clamp));
            }

            int output = hsum_8x32(sum);
            int retVal = (output / QA + net.LayerBiases[outputBucket]) * OutputScale / QAB;
            //std::cout << "retVal: " << retVal << std::endl;
            return retVal;
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

            return { whiteIndex * SIMD_CHUNKS, blackIndex * SIMD_CHUNKS };
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

            return ((768 * KingBuckets[kingSq]) + ((pc ^ perspective) * ColorStride) + (pt * PieceStride) + (sq)) * HiddenSize;
        }


        //  https://stackoverflow.com/questions/63106143/simd-c-avx2-intrinsics-getting-sum-of-m256i-vector-with-16bit-integers
        static int32_t hsum_8x32(__m256i v)
        {
            // silly GCC uses a longer AXV512VL instruction if AVX512 is enabled :/
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));

            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);                  // 3-operand non-destructive AVX lets us save a byte without needing a movdqa
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));    // Swap the low two elements
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            return _mm_cvtsi128_si32(sum32);       // movd
        }

        static void SubSubAddAdd(const __m256i* src, __m256i* dst, const __m256i* sub1, const __m256i* sub2, const __m256i* add1, const __m256i* add2)
        {
            for (int i = 0; i < SIMD_CHUNKS; i++)
            {
                dst[i] = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(src[i], add1[i]), add2[i]), sub1[i]), sub2[i]);
            }
        }

        static void SubSubAdd(const __m256i* src, __m256i* dst, const __m256i* sub1, const __m256i* sub2, const __m256i* add1)
        {
            for (int i = 0; i < SIMD_CHUNKS; i++)
            {
                dst[i] = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(src[i], add1[i]), sub1[i]), sub2[i]);
            }
        }

        static void SubAdd(const __m256i* src, __m256i* dst, const __m256i* sub1, const __m256i* add1)
        {
            for (int i = 0; i < SIMD_CHUNKS; i++)
            {
                dst[i] = _mm256_sub_epi16(_mm256_add_epi16(src[i], add1[i]), sub1[i]);
            }
        }

        static void Add(const __m256i* src, __m256i* dst, const __m256i* add1)
        {
            for (int i = 0; i < SIMD_CHUNKS; i++)
                dst[i] = _mm256_add_epi16(src[i], add1[i]);
        }

        static void Sub(const __m256i* src, __m256i* dst, const __m256i* sub1)
        {
            for (int i = 0; i < SIMD_CHUNKS; i++)
                dst[i] = _mm256_sub_epi16(src[i], sub1[i]);
        }
    }

}
