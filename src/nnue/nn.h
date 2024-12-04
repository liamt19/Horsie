#pragma once

#ifndef NN_H
#define NN_H

#include "../position.h"
#include "../nnue/arch.h"

#include <array>
#include <immintrin.h>



namespace Horsie
{
    namespace NNUE {

        using vec_i8 = __m256i;
        using vec_i16 = __m256i;
        using vec_i32 = __m256i;
        using vec_ps32 = __m256;
        using v128i = __m128i;

        template <typename T, typename W, typename U>
        struct alignas(64) QuantisedNetworkBase {
            T FTWeights[INPUT_SIZE * L1_SIZE * INPUT_BUCKETS];
            T FTBiases[L1_SIZE];
            W L1Weights[L1_SIZE][OUTPUT_BUCKETS][L2_SIZE];
            U L1Biases[OUTPUT_BUCKETS][L2_SIZE];
            U L2Weights[L2_SIZE][OUTPUT_BUCKETS][L3_SIZE];
            U L2Biases[OUTPUT_BUCKETS][L3_SIZE];
            U L3Weights[L3_SIZE][OUTPUT_BUCKETS];
            U L3Biases[OUTPUT_BUCKETS];
        };
        using QuantisedNetwork = QuantisedNetworkBase<short, sbyte, float>;

        template <typename T, typename W, typename U>
        struct alignas(64) NetworkBase
        {
            std::array<T, INPUT_SIZE * L1_SIZE * INPUT_BUCKETS>          FTWeights;
            std::array<T, L1_SIZE>                                       FTBiases;
            std::array<std::array<W, L1_SIZE * L2_SIZE>, OUTPUT_BUCKETS> L1Weights;
            std::array<std::array<U, L2_SIZE          >, OUTPUT_BUCKETS> L1Biases;
            std::array<std::array<U, L2_SIZE * L3_SIZE>, OUTPUT_BUCKETS> L2Weights;
            std::array<std::array<U, L3_SIZE          >, OUTPUT_BUCKETS> L2Biases;
            std::array<std::array<U, L3_SIZE          >, OUTPUT_BUCKETS> L3Weights;
            std::array<U, OUTPUT_BUCKETS>                                L3Biases;
        };
        using Network = NetworkBase<short, sbyte, float>;


        struct NNZTable {
            __m128i Entries[256];
        };

        extern Network net;
        extern NNZTable nnzTable;
        extern int PermuteIndices[L1_SIZE / 2];

        void LoadNetwork(const std::string& name);
        static void SetupNNZ();
        static void PermuteFT(std::array<short, INPUT_SIZE* L1_SIZE* INPUT_BUCKETS>& ftWeights, std::array<short, L1_SIZE>& ftBiases);
        static void PermuteL1(sbyte l1Weights[L1_SIZE][OUTPUT_BUCKETS][L2_SIZE]);


        void RefreshAccumulator(Position& pos);
        void RefreshAccumulatorPerspectiveFull(Position& pos, int perspective);
        void RefreshAccumulatorPerspective(Position& pos, int perspective);


        void MakeMoveNN(Position& pos, Move m);
        void MakeNullMove(Position& pos);
        void UpdateSingle(Accumulator* prev, Accumulator* curr, int perspective);
        void ProcessUpdates(Position& pos);


        int GetEvaluation(Position& pos, int outputBucket);
        int GetEvaluation(Position& pos);
        static void ActivateFTSparse(short* us, short* them, sbyte* weights, float* biases, float* output);
        static void ActivateL1Sparse(sbyte* inputs, sbyte* weights, float* biases, float* output, ushort* nnzIndices, int nnzCount);
        static void ActivateL2(float* inputs, float* weights, float* biases, float* output);
        static void ActivateL3(float* inputs, float* weights, float bias, float& output);


        std::pair<int, int> FeatureIndex(int pc, int pt, int sq, int wk, int bk);
        int FeatureIndexSingle(int pc, int pt, int sq, int kingSq, int perspective);


        constexpr int KingBuckets[] = {
             0,  1,  2,  3, 19, 18, 17, 16,
             4,  5,  6,  7, 23, 22, 21, 20,
             8,  9, 10, 11, 27, 26, 25, 24,
             8,  9, 10, 11, 27, 26, 25, 24,
            12, 12, 13, 13, 29, 29, 28, 28,
            12, 12, 13, 13, 29, 29, 28, 28,
            14, 14, 15, 15, 31, 31, 30, 30,
            14, 14, 15, 15, 31, 31, 30, 30,
        };


        template<typename IntType>
        inline IntType read_little_endian(std::istream& stream) {
            IntType result;
            stream.read(reinterpret_cast<char*>(&result), sizeof(IntType));
            return result;
        }

        void ResetCaches(Position& pos);
        static int32_t hsum_8x32(__m256i v);
        inline float scuffed_chatgpt_hsum(const vec_ps32 v);

        static void Sub(const short* src, short* dst, const short* sub1);
        static void Add(const short* src, short* dst, const short* add1);
        static void SubAdd(const short* src, short* dst, const short* sub1, const short* add1);
        static void SubSubAdd(const short* src, short* dst, const short* sub1, const short* sub2, const short* add1);
        static void SubSubAddAdd(const short* src, short* dst, const short* sub1, const short* sub2, const short* add1, const short* add2);

        inline uint16_t vec_nnz_mask(const vec_i32 vec) { return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(vec, _mm256_setzero_si256()))); }
        inline vec_i32 vec_dpbusd_epi32(const vec_i32 sum, const vec_i8 vec0, const vec_i8 vec1) {
            const vec_i16 product16 = _mm256_maddubs_epi16(vec0, vec1);
            const vec_i32 product32 = _mm256_madd_epi16(product16, _mm256_set1_epi16(1));
            return _mm256_add_epi32(sum, product32);
        }
        inline vec_ps32 vec_mul_add_ps(const vec_ps32 vec0, const vec_ps32 vec1, const vec_ps32 vec2) { return _mm256_fmadd_ps(vec0, vec1, vec2); }
    }

}

#endif
