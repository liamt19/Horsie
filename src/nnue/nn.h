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

    template <typename T, typename W, typename U>
    struct alignas(64) QuantisedNetworkBase {
        T FTWeights[INPUT_SIZE * L1_SIZE * INPUT_BUCKETS];
        T FTBiases [L1_SIZE];
        W L1Weights[OUTPUT_BUCKETS][L2_SIZE][L1_SIZE];
        U L1Biases [OUTPUT_BUCKETS][L2_SIZE];
        U L2Weights[OUTPUT_BUCKETS][L3_SIZE][L2_SIZE];
        U L2Biases [OUTPUT_BUCKETS][L3_SIZE];
        U L3Weights[OUTPUT_BUCKETS][L3_SIZE];
        U L3Biases [OUTPUT_BUCKETS];
    };
    using QuantisedNetwork = QuantisedNetworkBase<i16, i8, float>;

    template <typename T, typename W, typename U>
    struct alignas(64) NetworkBase {
        std::array<T, INPUT_SIZE * L1_SIZE * INPUT_BUCKETS> FTWeights;
        std::array<T, L1_SIZE>                              FTBiases;
        Util::NDArray<W, OUTPUT_BUCKETS, L1_SIZE * L2_SIZE> L1Weights;
        Util::NDArray<U, OUTPUT_BUCKETS, L2_SIZE>           L1Biases;
        Util::NDArray<U, OUTPUT_BUCKETS, L2_SIZE * L3_SIZE> L2Weights;
        Util::NDArray<U, OUTPUT_BUCKETS, L3_SIZE>           L2Biases;
        Util::NDArray<U, OUTPUT_BUCKETS, L3_SIZE>           L3Weights;
        std::array<U, OUTPUT_BUCKETS>                       L3Biases;
    };
    using Network = NetworkBase<i16, i8, float>;


    struct alignas(64) NNZTable {
        __m128i Entries[256];
    };

    extern Network net;
    extern NNZTable nnzTable;

    bool IsCompressed(std::istream& stream);
    void LoadZSTD(std::istream& m_stream, std::byte* dst);
    void LoadNetwork(const std::string& name);
    static void SetupNNZ();
    static void PermuteFT(Span<i16> ftWeights, Span<i16> ftBiases);
    static void PermuteL1(i8 l1Weights[OUTPUT_BUCKETS][L2_SIZE][L1_SIZE]);

    i32 GetEvaluation(Position& pos, i32 outputBucket);
    i32 GetEvaluation(Position& pos);

    std::pair<i32, i32> FeatureIndex(i32 pc, i32 pt, i32 sq, i32 wk, i32 bk);
    i32 FeatureIndexSingle(i32 pc, i32 pt, i32 sq, i32 kingSq, i32 perspective);


    constexpr i32 KingBuckets[] = {
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
    };

    constexpr i32 BucketForPerspective(i32 ksq, i32 perspective) {
        return (KingBuckets[(ksq ^ (56 * perspective))]);
    }


    template<typename IntType>
    inline IntType read_little_endian(std::istream& stream) {
        IntType result;
        stream.read(reinterpret_cast<char*>(&result), sizeof(IntType));
        return result;
    }

    void ResetCaches(Position& pos);

    void Sub(const i16* src, i16* dst, const i16* sub1);
    void Add(const i16* src, i16* dst, const i16* add1);
    void SubAdd(const i16* src, i16* dst, const i16* sub1, const i16* add1);
    void SubSubAdd(const i16* src, i16* dst, const i16* sub1, const i16* sub2, const i16* add1);
    void SubSubAddAdd(const i16* src, i16* dst, const i16* sub1, const i16* sub2, const i16* add1, const i16* add2);


    constexpr i32 BestPermuteIndices[] = {
        108, 342, 463, 401, 355, 293, 151, 369, 425, 209, 96, 207, 317, 469, 159, 175,
        460, 7, 486, 192, 243, 190, 452, 482, 126, 277, 204, 138, 330, 258, 395, 282,
        320, 137, 431, 104, 368, 379, 468, 128, 185, 448, 385, 381, 260, 239, 121, 238,
        427, 383, 359, 227, 483, 75, 27, 202, 439, 125, 95, 256, 307, 31, 92, 110,
        218, 502, 363, 109, 70, 264, 360, 326, 149, 272, 442, 90, 221, 480, 329, 2,
        473, 146, 323, 205, 97, 35, 371, 67, 22, 477, 80, 410, 176, 162, 489, 215,
        111, 411, 407, 173, 134, 311, 21, 143, 449, 61, 34, 321, 324, 8, 50, 378,
        285, 17, 312, 183, 343, 252, 120, 387, 263, 509, 222, 331, 432, 101, 361, 398,
        254, 305, 122, 436, 376, 364, 446, 200, 220, 443, 208, 212, 242, 116, 348, 153,
        68, 211, 174, 373, 129, 414, 386, 337, 224, 161, 437, 182, 281, 105, 46, 462,
        357, 322, 77, 400, 198, 408, 43, 345, 447, 296, 396, 199, 66, 14, 292, 426,
        235, 295, 214, 250, 38, 193, 273, 115, 164, 346, 493, 504, 299, 404, 157, 94,
        365, 341, 444, 327, 306, 457, 85, 347, 119, 213, 169, 354, 289, 18, 356, 340,
        127, 10, 510, 48, 286, 423, 351, 382, 241, 45, 507, 156, 178, 409, 268, 194,
        16, 406, 313, 506, 503, 11, 44, 246, 83, 316, 229, 344, 234, 187, 429, 232,
        270, 500, 424, 147, 60, 29, 100, 188, 74, 84, 240, 139, 271, 36, 13, 459,
        498, 445, 247, 253, 82, 244, 91, 349, 451, 12, 28, 338, 166, 350, 33, 228,
        4, 279, 435, 467, 78, 422, 390, 267, 236, 389, 20, 251, 284, 490, 367, 377,
        478, 496, 106, 453, 63, 49, 57, 226, 225, 328, 413, 495, 314, 494, 165, 98,
        58, 114, 65, 72, 308, 366, 201, 54, 136, 332, 197, 266, 132, 492, 399, 41,
        245, 158, 454, 315, 144, 511, 301, 278, 59, 150, 416, 297, 304, 53, 333, 394,
        1, 397, 195, 81, 302, 180, 93, 113, 393, 438, 73, 55, 434, 141, 319, 140,
        259, 15, 230, 276, 300, 89, 474, 249, 191, 171, 392, 418, 403, 487, 64, 62,
        485, 172, 130, 40, 210, 9, 223, 491, 206, 217, 99, 124, 23, 274, 309, 479,
        265, 0, 152, 472, 87, 465, 441, 145, 163, 412, 298, 112, 5, 203, 219, 107,
        353, 135, 168, 488, 310, 508, 56, 3, 155, 475, 374, 470, 497, 417, 71, 375,
        102, 288, 179, 86, 131, 384, 160, 362, 51, 26, 481, 499, 133, 216, 19, 255,
        186, 24, 402, 283, 430, 148, 262, 415, 189, 79, 339, 52, 290, 257, 177, 318,
        428, 181, 231, 370, 380, 405, 388, 287, 440, 269, 303, 335, 30, 118, 42, 466,
        47, 455, 25, 372, 458, 196, 76, 237, 433, 69, 419, 154, 352, 421, 32, 456,
        248, 334, 142, 420, 275, 184, 471, 37, 170, 280, 358, 325, 464, 39, 501, 117,
        461, 294, 391, 484, 167, 103, 261, 6, 336, 123, 450, 88, 476, 291, 233, 505,
    };

    constexpr auto PermuteIndices = [] {
        std::array<i32, L1_PAIR_COUNT> arr{};

        for (i32 i = 0; i < L1_PAIR_COUNT; ++i) {
#if defined(NO_WEIGHT_PERMUTING)
            arr[i] = i;
#else
            arr[i] = BestPermuteIndices[i];
#endif
        }

        return arr;
    }();

}
