#pragma once

#ifndef NN_H
#define NN_H

#define NO_PERM 1
//#undef NO_PERM

#include "../position.h"
#include "../nnue/arch.h"

#include <array>
#include <span>
#include <immintrin.h>

template<typename T>
using Span = std::span<T>;

namespace Horsie
{
    namespace NNUE {

        using vec_i8 = __m256i;
        using vec_i16 = __m256i;
        using vec_i32 = __m256i;
        using vec_ps32 = __m256;
        using vec_128i = __m128i;

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
        using QuantisedNetwork = QuantisedNetworkBase<i16, i8, float>;

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
        using Network = NetworkBase<i16, i8, float>;


        struct NNZTable {
            __m128i Entries[256];
        };

        extern Network net;
        extern NNZTable nnzTable;

        void LoadNetwork(const std::string& name);
        static void SetupNNZ();
        static void PermuteFT(Span<i16> ftWeights, Span<i16> ftBiases);
        static void PermuteL1(i8 l1Weights[L1_SIZE][OUTPUT_BUCKETS][L2_SIZE]);


        void RefreshAccumulator(Position& pos);
        void RefreshAccumulatorPerspectiveFull(Position& pos, i32 perspective);
        void RefreshAccumulatorPerspective(Position& pos, i32 perspective);


        void MakeMoveNN(Position& pos, Move m);
        void MakeNullMove(Position& pos);
        void UpdateSingle(Accumulator* prev, Accumulator* curr, i32 perspective);
        void ProcessUpdates(Position& pos);


        i32 GetEvaluation(Position& pos, i32 outputBucket);
        i32 GetEvaluation(Position& pos);
        static void ActivateFTSparse(Span<i16> us, Span<i16> them, Span<i8> weights, Span<float> biases, Span<float> output);
        static void ActivateL1Sparse(Span<i8> inputs, Span<i8> weights, Span<float> biases, Span<float> output, Span<u16> nnzIndices, const i32 nnzCount);
        static void ActivateL2(Span<float> inputs, Span<float> weights, Span<float> biases, Span<float> output);
        static void ActivateL3(Span<float> inputs, Span<float> weights, const float bias, float& output);


        std::pair<i32, i32> FeatureIndex(i32 pc, i32 pt, i32 sq, i32 wk, i32 bk);
        i32 FeatureIndexSingle(i32 pc, i32 pt, i32 sq, i32 kingSq, i32 perspective);


        constexpr i32 KingBuckets[] = {
             0,  1,  2,  3, 17, 16, 15, 14,
             4,  5,  6,  7, 21, 20, 19, 18,
             8,  9, 10, 11, 25, 24, 23, 22,
             8,  9, 10, 11, 25, 24, 23, 22,
            12, 12, 13, 13, 27, 27, 26, 26,
            12, 12, 13, 13, 27, 27, 26, 26,
            12, 12, 13, 13, 27, 27, 26, 26,
            12, 12, 13, 13, 27, 27, 26, 26,
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

        static void Sub(const i16* src, i16* dst, const i16* sub1);
        static void Add(const i16* src, i16* dst, const i16* add1);
        static void SubAdd(const i16* src, i16* dst, const i16* sub1, const i16* add1);
        static void SubSubAdd(const i16* src, i16* dst, const i16* sub1, const i16* sub2, const i16* add1);
        static void SubSubAddAdd(const i16* src, i16* dst, const i16* sub1, const i16* sub2, const i16* add1, const i16* add2);


        //  https://stackoverflow.com/questions/63106143/simd-c-avx2-intrinsics-getting-sum-of-m256i-vector-with-16bit-integers
        inline int32_t hsum_8x32(__m256i v) {
            // silly GCC uses a longer AXV512VL instruction if AVX512 is enabled :/
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));

            __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
            __m128i sum64 = _mm_add_epi32(hi64, sum128);
            __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
            __m128i sum32 = _mm_add_epi32(sum64, hi32);
            return _mm_cvtsi128_si32(sum32);
        }

        inline float scuffed_chatgpt_hsum(const vec_ps32 v) {
            __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            return _mm_cvtss_f32(sum128);
        }

        inline uint16_t vec_nnz_mask(const vec_i32 vec) { return _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(vec, _mm256_setzero_si256()))); }

        inline vec_i32 vec_dpbusd_epi32(const vec_i32 sum, const vec_i8 vec0, const vec_i8 vec1) {
            const vec_i16 product16 = _mm256_maddubs_epi16(vec0, vec1);
            const vec_i32 product32 = _mm256_madd_epi16(product16, _mm256_set1_epi16(1));
            return _mm256_add_epi32(sum, product32);
        }

        inline vec_ps32 vec_mul_add_ps(const vec_ps32 vec0, const vec_ps32 vec1, const vec_ps32 vec2) { return _mm256_fmadd_ps(vec0, vec1, vec2); }



        constexpr i32 BestPermuteIndices[] = {
            244, 462, 364, 595, 521, 53, 35, 406, 400, 291, 382, 372, 392, 606, 95, 569,
            240, 27, 108, 713, 155, 446, 405, 386, 567, 449, 270, 54, 43, 642, 146, 409,
            709, 201, 596, 131, 180, 361, 262, 417, 167, 616, 700, 169, 363, 517, 388, 132,
            113, 25, 285, 105, 226, 204, 580, 362, 15, 503, 753, 726, 494, 17, 659, 448,
            633, 207, 421, 345, 139, 743, 669, 247, 179, 331, 166, 72, 766, 215, 450, 199,
            545, 9, 261, 249, 558, 614, 589, 411, 103, 685, 650, 601, 573, 98, 41, 748,
            675, 617, 181, 3, 116, 106, 351, 173, 610, 658, 447, 348, 692, 565, 327, 667,
            661, 77, 398, 704, 310, 366, 623, 622, 122, 323, 324, 551, 668, 57, 170, 92,
            432, 237, 578, 350, 638, 258, 387, 96, 460, 86, 593, 452, 518, 47, 4, 87,
            742, 629, 583, 458, 693, 690, 265, 710, 346, 293, 64, 396, 402, 370, 198, 630,
            729, 496, 299, 620, 491, 60, 280, 465, 128, 651, 150, 597, 75, 235, 469, 456,
            532, 384, 118, 542, 625, 151, 618, 295, 423, 102, 393, 297, 754, 318, 335, 76,
            682, 28, 30, 277, 702, 533, 130, 234, 129, 250, 546, 301, 719, 22, 724, 56,
            238, 302, 12, 23, 153, 253, 367, 588, 640, 478, 154, 340, 440, 752, 676, 334,
            598, 422, 413, 486, 435, 699, 242, 136, 637, 133, 514, 7, 536, 31, 490, 498,
            581, 605, 548, 631, 48, 99, 390, 463, 572, 767, 371, 80, 632, 337, 187, 1,
            110, 176, 734, 36, 665, 10, 44, 368, 434, 263, 391, 603, 579, 191, 747, 485,
            352, 115, 722, 761, 188, 627, 407, 79, 55, 369, 554, 120, 652, 227, 681, 91,
            511, 397, 93, 356, 575, 109, 559, 714, 142, 677, 584, 212, 539, 14, 147, 733,
            635, 189, 430, 523, 750, 254, 29, 190, 720, 433, 309, 279, 427, 708, 148, 174,
            522, 159, 88, 339, 288, 67, 246, 657, 666, 61, 305, 168, 365, 39, 62, 193,
            439, 731, 504, 257, 760, 117, 510, 40, 332, 582, 574, 225, 401, 441, 561, 205,
            66, 338, 639, 751, 186, 718, 221, 85, 165, 497, 8, 464, 46, 416, 739, 684,
            359, 544, 473, 274, 660, 196, 49, 267, 738, 59, 557, 624, 765, 200, 443, 282,
            746, 461, 541, 385, 568, 643, 468, 38, 94, 152, 278, 178, 358, 538, 647, 264,
            556, 671, 33, 241, 163, 20, 298, 355, 222, 42, 442, 549, 292, 16, 563, 537,
            255, 686, 459, 515, 646, 303, 239, 177, 329, 506, 749, 260, 320, 101, 149, 426,
            172, 90, 223, 217, 357, 599, 233, 314, 330, 213, 477, 218, 707, 231, 673, 721,
            655, 0, 347, 197, 184, 290, 474, 653, 328, 600, 50, 648, 6, 162, 663, 408,
            5, 641, 737, 756, 275, 475, 695, 609, 317, 527, 157, 266, 755, 81, 203, 487,
            394, 286, 586, 403, 380, 209, 195, 414, 571, 706, 672, 495, 698, 566, 730, 294,
            429, 287, 736, 534, 374, 68, 697, 687, 520, 343, 322, 425, 336, 705, 89, 513,
            644, 489, 516, 354, 723, 703, 428, 757, 58, 100, 333, 525, 307, 342, 360, 19,
            483, 236, 283, 112, 37, 107, 229, 611, 628, 127, 276, 577, 519, 732, 379, 321,
            501, 206, 73, 272, 608, 245, 418, 194, 353, 472, 125, 208, 316, 712, 570, 210,
            564, 304, 268, 373, 455, 654, 492, 104, 481, 562, 488, 502, 499, 696, 550, 134,
            741, 381, 484, 389, 383, 70, 530, 243, 735, 164, 528, 466, 535, 560, 52, 399,
            500, 759, 377, 300, 547, 308, 727, 161, 694, 126, 269, 649, 2, 524, 18, 137,
            21, 679, 124, 678, 121, 656, 220, 412, 315, 445, 135, 34, 69, 312, 140, 431,
            626, 437, 590, 505, 141, 470, 281, 84, 674, 182, 349, 612, 375, 256, 691, 71,
            725, 82, 273, 607, 341, 587, 670, 444, 78, 65, 325, 436, 160, 395, 45, 230,
            296, 83, 32, 97, 688, 540, 613, 111, 740, 145, 457, 716, 232, 509, 192, 219,
            175, 764, 479, 602, 591, 319, 453, 482, 51, 526, 138, 248, 683, 553, 711, 493,
            11, 555, 344, 645, 531, 576, 251, 211, 438, 419, 664, 311, 471, 74, 715, 259,
            758, 26, 404, 306, 271, 183, 185, 680, 701, 507, 424, 158, 143, 313, 156, 123,
            508, 119, 451, 604, 621, 476, 467, 529, 13, 512, 228, 585, 224, 289, 410, 762,
            744, 171, 763, 214, 420, 745, 454, 144, 662, 594, 480, 252, 634, 202, 717, 326,
            689, 63, 24, 114, 216, 284, 376, 378, 415, 543, 552, 592, 615, 619, 636, 728,
        };

        constexpr auto PermuteIndices = [] 
        {
            std::array<i32, L1_PAIR_COUNT> arr{};
            
            for (i32 i = 0; i < L1_PAIR_COUNT; ++i) {
#if defined(NO_PERM)
                arr[i] = i;
#else
                arr[i] = BestPermuteIndices[i];
#endif
            }

            return arr;
        }();

    }

}

#endif
