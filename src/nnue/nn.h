#pragma once

#define PERM_COUNT 1
#undef PERM_COUNT

#define NO_WEIGHT_PERMUTING 1
#undef NO_WEIGHT_PERMUTING

#include "../defs.h"
#include "../nnue/arch.h"
#include "../position.h"
#include "accumulator.h"

#include <array>
#include <span>

namespace Horsie::NNUE {

#if defined(PERM_COUNT)
    extern std::array<u64, L1_SIZE> NNZCounts;
    extern u64 ActivationCount;
    extern u64 EvalCalls;
#endif

    template<typename T>
    using Span = std::span<T>;

    struct alignas(64) Network {
        Util::NDArray<float, OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE> Conv1Weights;
        Util::NDArray<float, OUT_CHANNELS>                                        Conv1Biases;
        Util::NDArray<float, SQUARE_NB, OUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE>   Conv2Weights;
        Util::NDArray<float, SQUARE_NB>                                           Conv2Biases;
        Util::NDArray<float, FC_LAYER_SIZE, FC_LAYER_SIZE * OUT_CHANNELS>         FC1Weights;
        Util::NDArray<float, FC_LAYER_SIZE>                                       FC1Biases;
        Util::NDArray<float, FC_LAYER_SIZE>                                       FC2Weights;
        Util::NDArray<float, 1>                                                   FC2Biases;
    };

    extern Network net;

    bool IsCompressed(std::istream& stream);
    void LoadZSTD(std::istream& m_stream, std::byte* dst);
    void LoadNetwork(const std::string& name);

    i32 GetEvaluation(const Position& pos);

    template<typename IntType>
    inline IntType read_little_endian(std::istream& stream) {
        IntType result;
        stream.read(reinterpret_cast<char*>(&result), sizeof(IntType));
        return result;
    }

    void ResetCaches(Position& pos);

    template<i32 in_channels, i32 out_channels>
    struct Conv2D {
        float* weight;
        float* bias;
        Conv2D(float* w, float* b) : weight(w), bias(b) {}

        void forward__(const float* in, float* out) const {
            constexpr int H = ROW_LEN, W = ROW_LEN;
            constexpr int K = KERNEL_SIZE;
            constexpr int pad = KERNEL_PADDING;

            for (int oc = 0; oc < out_channels; oc++) {
                const auto outChannelBlock = oc * (in_channels * K * K);

                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {

                        float sum = bias[oc];

                        for (int ic = 0; ic < in_channels; ic++) {
                            const auto inChannelBlock = ic * (K * K);

                            for (int ky = 0; ky < K; ky++) {
                                for (int kx = 0; kx < K; kx++) {

                                    int iy = y + ky - pad;
                                    int ix = x + kx - pad;

                                    if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                                        continue;

                                    const auto kernelIndex = ky * K + kx;
                                    float w = weight[outChannelBlock + inChannelBlock + kernelIndex];

                                    float v = in[ic * (H * W) + iy * W + ix];

                                    sum += w * v;
                                }
                            }
                        }

                        out[oc * (H * W) + y * W + x] = std::clamp(sum, 0.0f, 1.0f);
                    }
                }
            }
        }

        void forward(const float* input, float* output) const {
            constexpr int H = 8;
            constexpr int W = 8;
            constexpr int HW = H * W;      // 64
            constexpr int K = 3;
            constexpr int KK = K * K;      // 9

            const int rows = in_channels * KK;           // im2col rows
            const int cols = HW;                         // im2col cols (64)

            // Build im2col: rows x cols
            // im2col[r * cols + s] where r = ic*9 + ky*3 + kx, s = y*8 + x
            std::array<float, rows * cols> im2col{};
            auto im = &im2col[0];

            // Fill im2col (zero for padding)
            for (int ic = 0; ic < in_channels; ++ic) {
                const float* in_chan = input + ic * HW; // pointer to channel data (64 floats)
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        int r = ic * KK + ky * K + kx;
                        float* im_row = im + r * cols;

                        // For each output spatial position s, compute input y/x
                        for (int s = 0; s < cols; ++s) {
                            int y = s / W;
                            int x = s % W;
                            int iy = y + ky - 1; // pad = 1
                            int ix = x + kx - 1;

                            float v = 0.0f;
                            if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W) {
                                v = in_chan[iy * W + ix];
                            }
                            im_row[s] = v;
                        }
                    }
                }
            }

            // Vectorized accumulation:
            // Process spatial cols in blocks of 8 (since AVX2 = 8 floats)
            const int step = 8;
            const int nBlocks = cols / step; // 64 / 8 = 8 blocks; exact division

            for (int oc = 0; oc < out_channels; ++oc) {
                const float* w_oc = weight + oc * (in_channels * KK); // weights for this output channel
                const float b = bias ? bias[oc] : 0.0f;

                for (int block = 0; block < nBlocks; ++block) {
                    int s0 = block * step;
                    // initialize sum vector with bias
                    __m256 sum = _mm256_set1_ps(b);

                    // accumulate sum += w_oc[k] * im[k][s0..s0+7]
                    for (int k = 0; k < rows; ++k) {
                        // load 8 floats from im2col row k at column s0
                        __m256 vec = _mm256_loadu_ps(im + k * cols + s0);
                        __m256 w_bcast = _mm256_set1_ps(w_oc[k]);

                        sum = _mm256_fmadd_ps(w_bcast, vec, sum);
                    }

                    sum = _mm256_min_ps(_mm256_max_ps(sum, _mm256_setzero_ps()), _mm256_set1_ps(1.0f));

                    // store result into output[oc * HW + s0 .. s0+7]
                    _mm256_storeu_ps(output + oc * HW + s0, sum);
                }
            }
        }
    };

    template<i32 in_features, i32 out_features, bool isOutput>
    struct Linear {
        float* weight;
        float* bias;
        Linear(float* w, float* b) : weight(w), bias(b) {}

        void forward(const float* in, float* out) const {
            for (int o = 0; o < out_features; o++) {
                float sum = bias[o];
                const float* wrow = &weight[o * in_features];

                for (int i = 0; i < in_features; i++)
                    sum += wrow[i] * in[i];

                out[o] = isOutput ? sum : std::clamp(sum, 0.0f, 1.0f);
            }
        }
    };

}
