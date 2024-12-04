#pragma once

#include <immintrin.h>

constexpr int INPUT_BUCKETS = 16;
constexpr int INPUT_SIZE = 768;
constexpr int L1_SIZE = 1536;
constexpr int L2_SIZE = 16;
constexpr int L3_SIZE = 32;
constexpr int OUTPUT_BUCKETS = 8;

constexpr int FT_QUANT = 255;
constexpr int FT_SHIFT = 10;
constexpr int L1_QUANT = 64;
constexpr int OutputScale = 400;

constexpr int U8_CHUNK_SIZE = sizeof(__m256i) / sizeof(byte);
constexpr int I16_CHUNK_SIZE = sizeof(__m256i) / sizeof(short);
constexpr int I32_CHUNK_SIZE = sizeof(__m256i) / sizeof(int);
constexpr int F32_CHUNK_SIZE = sizeof(__m256i) / sizeof(float);

constexpr int NNZ_INPUT_SIMD_WIDTH = sizeof(__m256i) / sizeof(int);
constexpr int NNZ_CHUNK_SIZE = (NNZ_INPUT_SIMD_WIDTH > 8) ? NNZ_INPUT_SIMD_WIDTH : 8;
constexpr int NNZ_OUTPUTS_PER_CHUNK = NNZ_CHUNK_SIZE / 8;

constexpr int L1_CHUNK_PER_32 = sizeof(int) / sizeof(sbyte);
constexpr int L1_PAIR_COUNT = L1_SIZE / 2;

constexpr int SIMD_CHUNKS = L1_SIZE / (sizeof(__m256i) / sizeof(short));

constexpr int N_FTW = INPUT_SIZE * L1_SIZE * INPUT_BUCKETS;
constexpr int N_FTB = L1_SIZE;
constexpr int N_L1W = OUTPUT_BUCKETS * L1_SIZE * L2_SIZE;
constexpr int N_L1B = OUTPUT_BUCKETS * L2_SIZE;
constexpr int N_L2W = OUTPUT_BUCKETS * L2_SIZE * L3_SIZE;
constexpr int N_L2B = OUTPUT_BUCKETS * L3_SIZE;
constexpr int N_L3W = OUTPUT_BUCKETS * L3_SIZE;
constexpr int N_L3B = OUTPUT_BUCKETS;