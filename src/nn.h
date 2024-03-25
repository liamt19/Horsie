#pragma once

#ifndef NN_H
#define NN_H

#include "position.h"

#include <array>
#include <immintrin.h>

namespace Horsie
{
    namespace NNUE {

        const int InputSize = 768;
        const int HiddenSize = 1536;
        const int OutputBuckets = 1;

        const int QA = 255;
        const int QB = 64;
        const int QAB = QA * QB;

        const int OutputScale = 400;
        const bool SelectOutputBucket = (OutputBuckets != 1);

        const int SIMD_CHUNKS = HiddenSize / (sizeof(__m256i) / sizeof(short));

        const int FeatureWeightElements = InputSize * HiddenSize;
        const int FeatureBiasElements = HiddenSize;
        const int LayerWeightElements = HiddenSize * 2 * OutputBuckets;
        const int LayerBiasElements = OutputBuckets;

        struct alignas(64) Network
        {
            std::array<short, FeatureWeightElements> FeatureWeights;
            std::array<short, FeatureBiasElements>   FeatureBiases;
            std::array<short, LayerWeightElements>   LayerWeights;
            std::array<short, LayerBiasElements>     LayerBiases;
        };

        extern const Network& g_network;

        void LoadNetwork(const std::string& name);

        void RefreshAccumulator(Position& pos);
        void MakeMoveNN(Position& pos, Move m);
        int GetEvaluation(Position& pos);

        std::pair<int, int> FeatureIndex(int pc, int pt, int sq);


        void init_tables();
        int Pesto(Position& pos);
    }

}

#endif
