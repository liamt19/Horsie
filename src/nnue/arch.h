#pragma once

#include <immintrin.h>
#include "../defs.h"

constexpr auto InputBuckets = 16;
constexpr auto InputSize = 768;
constexpr auto HiddenSize = 2048;
constexpr auto OutputBuckets = 8;

constexpr auto QA = 258;
constexpr auto QB = 64;
constexpr auto QAB = QA * QB;

constexpr auto OutputScale = 400;

constexpr auto SIMD_CHUNKS = HiddenSize / (sizeof(__m256i) / sizeof(i16));

const auto FeatureWeightElements = InputSize * HiddenSize;
const auto FeatureBiasElements = HiddenSize;
const auto LayerWeightElements = HiddenSize;
const auto LayerBiasElements = OutputBuckets;