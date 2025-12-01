#pragma once

#include "../defs.h"
#include "simd.h"

namespace Horsie::NNUE {
    constexpr auto INPUT_BUCKETS = 1;
    constexpr auto INPUT_SIZE = 768;
    constexpr auto IN_CHANNELS = 6 * 2;
    constexpr auto OUT_CHANNELS = 32;
    constexpr auto ROW_LEN = 8;
    constexpr auto KERNEL_SIZE = 3;
    constexpr auto KERNEL_PADDING = 1;
    constexpr auto FC_LAYER_SIZE = 128;
    constexpr auto OUTPUT_BUCKETS = 1;

    constexpr auto FC1_INPUTS = 64 * ROW_LEN * ROW_LEN;
    constexpr auto CONV1_EXTENDED_SIZE = OUT_CHANNELS * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;

    constexpr auto OutputScale = 400;

    constexpr auto U8_CHUNK_SIZE = sizeof(vec_i8) / sizeof(u8);
    constexpr auto I16_CHUNK_SIZE = sizeof(vec_i16) / sizeof(i16);
    constexpr auto I32_CHUNK_SIZE = sizeof(vec_i32) / sizeof(i32);
    constexpr auto F32_CHUNK_SIZE = sizeof(vec_ps) / sizeof(float);

}