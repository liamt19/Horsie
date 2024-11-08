#pragma once

#include <immintrin.h>

constexpr int InputBuckets = 5;
constexpr int InputSize = 768;
constexpr int HiddenSize = 1536;
constexpr int OutputBuckets = 8;

constexpr int QA = 258;
constexpr int QB = 64;
constexpr int QAB = QA * QB;

constexpr int OutputScale = 400;
constexpr bool SelectOutputBucket = (OutputBuckets != 1);

constexpr int SIMD_CHUNKS = HiddenSize / (sizeof(__m256i) / sizeof(short));

const int FeatureWeightElements = InputSize * HiddenSize;
const int FeatureBiasElements = HiddenSize;
const int LayerWeightElements = HiddenSize * 2;
const int LayerBiasElements = OutputBuckets;