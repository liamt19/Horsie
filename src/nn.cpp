
#define INCBIN_PREFIX g_
#include "./incbin/incbin.h"

#include "nn.h"
#include "position.h"
#include "immintrin.h"

#include <fstream>
#include <cstring>

#if defined(_MSC_VER)
#include "incbin/incbin_at_home.h"
#endif

namespace Horsie
{

    namespace NNUE {

        namespace {
#if defined(_MSC_VER)
            //Network s_network{};
            const Network* s_network = reinterpret_cast<const Network*>(incbin_rawData);
#else
            INCBIN(net, NETWORK_FILE);
            const Network* s_network = reinterpret_cast<const Network*>(g_netData);
#endif
        }

        const Network& g_network = *s_network;

        static int32_t hsum_8x32(__m256i v);
        static void SubAdd(__m256i* src, const __m256i* sub1, const __m256i* add1);
        static void SubSubAdd(__m256i* src, const __m256i* sub1, const __m256i* sub2, const __m256i* add1);
        static void SubSubAddAdd(__m256i* src, const __m256i* sub1, const __m256i* sub2, const __m256i* add1, const __m256i* add2);




        void RefreshAccumulator(Position& pos) {
            Accumulator& accumulator = *pos.State->Accumulator;
            Bitboard& bb = pos.bb;

#if NO
            for (size_t i = 0; i < HiddenSize; i++)
            {
                accumulator.White[i] = g_network.FeatureBiases[i];
                accumulator.Black[i] = g_network.FeatureBiases[i];
            }
#else
            std::memcpy(&accumulator.White[0], &g_network.FeatureBiases[0], HiddenSize * sizeof(short));
            std::memcpy(&accumulator.Black[0], &g_network.FeatureBiases[0], HiddenSize * sizeof(short));
#endif

            ulong occ = bb.Occupancy;
            while (occ != 0)
            {
                int pieceIdx = poplsb(occ);

                int pt = bb.GetPieceAtIndex(pieceIdx);
                int pc = bb.GetColorAtIndex(pieceIdx);

                const auto [wIdx, bIdx] = FeatureIndex(pc, pt, pieceIdx);

#if NO
                for (int i = 0; i < HiddenSize; i++)
                {
                    accumulator.White[i] = accumulator.White[i] + g_network.FeatureWeights[(wIdx / SIMD_CHUNKS) + i];
                    accumulator.Black[i] = accumulator.Black[i] + g_network.FeatureWeights[(bIdx / SIMD_CHUNKS) + i];
                }
#else
                auto white = reinterpret_cast<__m256i*>(accumulator.White);
                auto black = reinterpret_cast<__m256i*>(accumulator.Black);
                auto weights = reinterpret_cast<const __m256i*>(&g_network.FeatureWeights[0]);
                for (int i = 0; i < SIMD_CHUNKS; i++)
                {
                    _mm256_store_si256(&white[i], _mm256_add_epi16(_mm256_load_si256(&white[i]), _mm256_load_si256(&weights[wIdx + i])));
                    _mm256_store_si256(&black[i], _mm256_add_epi16(_mm256_load_si256(&black[i]), _mm256_load_si256(&weights[bIdx + i])));
                }
#endif

            }
        }



        void MakeMoveNN(Position& pos, Move m) {
            Bitboard bb = pos.bb;

            Accumulator* accumulator = pos.NextState()->Accumulator;
            pos.State->Accumulator->CopyTo(accumulator);

            int moveTo = m.To();
            int moveFrom = m.From();

            int us = pos.ToMove;
            int ourPiece = bb.GetPieceAtIndex(moveFrom);

            int them = Not(us);
            int theirPiece = bb.GetPieceAtIndex(moveTo);

            auto whiteAccumulation = reinterpret_cast<__m256i*>((*accumulator)[WHITE]);
            auto blackAccumulation = reinterpret_cast<__m256i*>((*accumulator)[BLACK]);

            const auto [wFrom, bFrom] = FeatureIndex(us, ourPiece, moveFrom);
            const auto [wTo, bTo] = FeatureIndex(us, m.IsPromotion() ? m.PromotionTo() : ourPiece, moveTo);

            auto FeatureWeights = reinterpret_cast<const __m256i*>(&g_network.FeatureWeights[0]);

            if (m.IsCastle())
            {
                int rookFrom = moveTo;
                int rookTo = m.CastlingRookSquare();

                const auto [wToC, bToC] = FeatureIndex(us, ourPiece, m.CastlingKingSquare());

                const auto [wRookFrom, bRookFrom] = FeatureIndex(us, ROOK, rookFrom);
                const auto [wRookTo, bRookTo] = FeatureIndex(us, ROOK, rookTo);

                SubSubAddAdd(whiteAccumulation,
                    (FeatureWeights + wFrom),
                    (FeatureWeights + wRookFrom),
                    (FeatureWeights + wToC),
                    (FeatureWeights + wRookTo));

                SubSubAddAdd(blackAccumulation,
                    (FeatureWeights + bFrom),
                    (FeatureWeights + bRookFrom),
                    (FeatureWeights + bToC),
                    (FeatureWeights + bRookTo));
            }
            else if (theirPiece != Piece::NONE)
            {
                const auto [wCap, bCap] = FeatureIndex(them, theirPiece, moveTo);

                SubSubAdd(whiteAccumulation,
                    (FeatureWeights + wFrom),
                    (FeatureWeights + wCap),
                    (FeatureWeights + wTo));

                SubSubAdd(blackAccumulation,
                    (FeatureWeights + bFrom),
                    (FeatureWeights + bCap),
                    (FeatureWeights + bTo));
            }
            else if (m.IsEnPassant())
            {
                int idxPawn = moveTo - ShiftUpDir(us);

                const auto [wCap, bCap] = FeatureIndex(them, PAWN, idxPawn);

                SubSubAdd(whiteAccumulation,
                    (FeatureWeights + wFrom),
                    (FeatureWeights + wCap),
                    (FeatureWeights + wTo));

                SubSubAdd(blackAccumulation,
                    (FeatureWeights + bFrom),
                    (FeatureWeights + bCap),
                    (FeatureWeights + bTo));
            }
            else
            {
                SubAdd(whiteAccumulation,
                    (FeatureWeights + wFrom),
                    (FeatureWeights + wTo));

                SubAdd(blackAccumulation,
                    (FeatureWeights + bFrom),
                    (FeatureWeights + bTo));
            }
        }



        int GetEvaluation(Position& pos) {

            Accumulator& accumulator = *pos.State->Accumulator;
            Bitboard& bb = pos.bb;
            
            const __m256i ClampMin = _mm256_setzero_si256();
            const __m256i ClampMax = _mm256_set1_epi16(QA);
            __m256i sum = _mm256_setzero_si256();

            auto  stm = reinterpret_cast<const __m256i*>(accumulator[pos.ToMove]);
            auto nstm = reinterpret_cast<const __m256i*>(accumulator[Not(pos.ToMove)]);
            auto layerWeights = reinterpret_cast<const __m256i*>(&g_network.LayerWeights[0]);

            for (int i = 0; i < SIMD_CHUNKS; i++) {
                __m256i clamp = _mm256_min_epi16(ClampMax, _mm256_max_epi16(ClampMin, stm[i]));
                __m256i mult = _mm256_mullo_epi16(clamp, layerWeights[i]);
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(mult, clamp));
            }

            for (int i = 0; i < SIMD_CHUNKS; i++) {
                __m256i clamp = _mm256_min_epi16(ClampMax, _mm256_max_epi16(ClampMin, nstm[i]));
                __m256i mult = _mm256_mullo_epi16(clamp, layerWeights[i + SIMD_CHUNKS]);
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(mult, clamp));
            }

            int output = hsum_8x32(sum);
            int retVal = (output / QA + g_network.LayerBiases[0]) * OutputScale / QAB;
            //std::cout << "retVal: " << retVal << std::endl;
            return retVal;
        }



        std::pair<int, int> FeatureIndex(int pc, int pt, int sq)
        {
            const int ColorStride = 64 * 6;
            const int PieceStride = 64;

            int whiteIndex = pc * ColorStride + pt * PieceStride + sq;
            int blackIndex = Not(pc) * ColorStride + pt * PieceStride + (sq ^ 56);

            return { whiteIndex * SIMD_CHUNKS, blackIndex * SIMD_CHUNKS };
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

        static void SubSubAddAdd(__m256i* src, const __m256i* sub1, const __m256i* sub2, const __m256i* add1, const __m256i* add2)
        {
            for (int i = 0; i < SIMD_CHUNKS; i++)
            {
                src[i] = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(src[i], add1[i]), add2[i]), sub1[i]), sub2[i]);
            }
        }

        static void SubSubAdd(__m256i* src, const __m256i* sub1, const __m256i* sub2, const __m256i* add1)
        {
            for (int i = 0; i < SIMD_CHUNKS; i++)
            {
                src[i] = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(src[i], add1[i]), sub1[i]), sub2[i]);
            }
        }

        static void SubAdd(__m256i* src, const __m256i* sub1, const __m256i* add1)
        {
            for (int i = 0; i < SIMD_CHUNKS; i++)
            {
                src[i] = _mm256_sub_epi16(_mm256_add_epi16(src[i], add1[i]), sub1[i]);
            }
        }
    }

}
