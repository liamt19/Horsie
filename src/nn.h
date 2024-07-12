#pragma once

#ifndef NN_H
#define NN_H

#include "position.h"

#include <array>
#include <immintrin.h>



namespace Horsie
{
    namespace NNUE {

        constexpr int InputBuckets = 5;
        constexpr int InputSize = 768;
        constexpr int HiddenSize = 1536;
        constexpr int OutputBuckets = 8;

        const int QA = 258;
        const int QB = 64;
        const int QAB = QA * QB;

        const int OutputScale = 400;
        constexpr bool SelectOutputBucket = (OutputBuckets != 1);

        const int SIMD_CHUNKS = HiddenSize / (sizeof(__m256i) / sizeof(short));

        const int FeatureWeightElements = InputSize * HiddenSize;
        const int FeatureBiasElements = HiddenSize;
        const int LayerWeightElements = HiddenSize * 2;
        const int LayerBiasElements = OutputBuckets;

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
            0, 0, 1, 1, 6, 6, 5, 5,
            2, 2, 3, 3, 8, 8, 7, 7,
            4, 4, 4, 4, 9, 9, 9, 9,
            4, 4, 4, 4, 9, 9, 9, 9,
            4, 4, 4, 4, 9, 9, 9, 9,
            4, 4, 4, 4, 9, 9, 9, 9,
            4, 4, 4, 4, 9, 9, 9, 9,
            4, 4, 4, 4, 9, 9, 9, 9,
        };


        template<typename IntType>
        inline IntType read_little_endian(std::istream& stream) {
            IntType result;
            stream.read(reinterpret_cast<char*>(&result), sizeof(IntType));
            return result;
        }

    }

}

#endif
