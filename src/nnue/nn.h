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

        struct alignas(64) Network
        {
            std::array<short, FeatureWeightElements * InputBuckets>             FeatureWeights;
            std::array<short, FeatureBiasElements>                              FeatureBiases;
            std::array<std::array<short, LayerWeightElements>, OutputBuckets>   LayerWeights;
            std::array<short, LayerBiasElements>                                LayerBiases;
        };

        extern Network net;

        void LoadNetwork(const std::string& name);

        void RefreshAccumulator(Position& pos);
        void RefreshAccumulatorPerspectiveFull(Position& pos, int perspective);
        void RefreshAccumulatorPerspective(Position& pos, int perspective);

        void MakeMoveNN(Position& pos, Move m);
        int GetEvaluation(Position& pos);

        std::pair<int, int> FeatureIndex(int pc, int pt, int sq, int wk, int bk);
        int FeatureIndexSingle(int pc, int pt, int sq, int kingSq, int perspective);


        constexpr int KingBuckets[] = {
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1,
            0, 0, 0, 0, 1, 1, 1, 1,
        };


        template<typename IntType>
        inline IntType read_little_endian(std::istream& stream) {
            IntType result;
            stream.read(reinterpret_cast<char*>(&result), sizeof(IntType));
            return result;
        }


        static int32_t hsum_8x32(__m256i v);
        static void Sub(const __m256i* src, __m256i* dst, const __m256i* sub1);
        static void Add(const __m256i* src, __m256i* dst, const __m256i* add1);
        static void SubAdd(const __m256i* src, __m256i* dst, const __m256i* sub1, const __m256i* add1);
        static void SubSubAdd(const __m256i* src, __m256i* dst, const __m256i* sub1, const __m256i* sub2, const __m256i* add1);
        static void SubSubAddAdd(const __m256i* src, __m256i* dst, const __m256i* sub1, const __m256i* sub2, const __m256i* add1, const __m256i* add2);


    }

}

#endif
