#pragma once

#ifndef NN_H
#define NN_H

#define PERM_COUNT 1
#undef PERM_COUNT

#define NO_PERM 1
#undef NO_PERM

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
            266, 174, 334, 369, 109, 501, 742, 125, 299, 130, 622, 727, 238, 355, 475, 696,
            173, 429, 535, 341, 479, 457, 508, 326, 285, 756, 252, 33, 385, 766, 123, 143,
            609, 415, 405, 218, 251, 547, 481, 0, 172, 507, 244, 596, 427, 304, 762, 118,
            430, 259, 692, 332, 19, 163, 305, 414, 359, 718, 152, 24, 336, 70, 63, 91,
            32, 724, 496, 181, 211, 445, 542, 395, 603, 347, 160, 482, 642, 714, 322, 570,
            578, 555, 364, 279, 446, 605, 765, 371, 3, 707, 316, 79, 131, 439, 660, 673,
            597, 497, 295, 504, 203, 2, 416, 503, 602, 48, 223, 135, 422, 471, 374, 249,
            155, 191, 350, 526, 532, 465, 413, 186, 46, 187, 198, 237, 733, 639, 637, 202,
            761, 264, 57, 466, 467, 394, 92, 477, 565, 688, 588, 272, 625, 286, 443, 682,
            358, 365, 199, 720, 62, 1, 368, 97, 86, 648, 438, 137, 227, 333, 460, 115,
            490, 401, 586, 573, 138, 260, 666, 582, 151, 182, 702, 554, 275, 708, 447, 480,
            495, 613, 119, 608, 153, 210, 212, 352, 661, 686, 221, 363, 650, 522, 630, 487,
            215, 737, 551, 618, 690, 240, 538, 659, 209, 469, 267, 744, 278, 168, 296, 462,
            653, 524, 121, 489, 99, 329, 589, 468, 571, 377, 28, 728, 233, 166, 428, 709,
            440, 546, 719, 472, 81, 320, 491, 247, 165, 486, 760, 587, 735, 127, 192, 20,
            748, 353, 674, 600, 361, 528, 117, 101, 594, 669, 14, 431, 302, 59, 274, 514,
            239, 66, 640, 235, 148, 83, 280, 411, 18, 93, 739, 432, 409, 590, 378, 577,
            310, 402, 206, 72, 529, 679, 348, 583, 498, 248, 67, 392, 398, 515, 423, 553,
            76, 752, 337, 224, 255, 50, 520, 662, 644, 100, 419, 116, 540, 723, 288, 236,
            339, 550, 473, 21, 317, 35, 736, 294, 351, 157, 699, 493, 229, 250, 633, 16,
            200, 283, 98, 634, 451, 303, 346, 606, 403, 426, 701, 26, 60, 41, 141, 54,
            767, 271, 615, 205, 455, 563, 89, 678, 717, 591, 731, 722, 258, 635, 95, 397,
            183, 243, 349, 84, 10, 193, 593, 113, 539, 478, 82, 537, 638, 25, 534, 214,
            437, 741, 170, 254, 171, 139, 372, 483, 763, 85, 393, 311, 743, 549, 308, 700,
            725, 400, 253, 523, 646, 492, 525, 390, 677, 234, 564, 598, 521, 291, 747, 162,
            511, 190, 753, 381, 738, 734, 159, 245, 631, 750, 531, 670, 470, 516, 194, 156,
            424, 435, 7, 327, 464, 277, 656, 22, 290, 325, 370, 541, 23, 106, 69, 453,
            456, 80, 56, 167, 201, 704, 180, 297, 17, 513, 420, 242, 675, 103, 196, 626,
            623, 219, 301, 706, 27, 379, 751, 179, 533, 387, 289, 145, 293, 518, 161, 629,
            344, 517, 154, 713, 510, 621, 384, 444, 222, 328, 746, 698, 40, 383, 755, 552,
            52, 417, 12, 133, 655, 544, 307, 11, 232, 108, 318, 281, 177, 55, 149, 144,
            37, 441, 617, 664, 335, 164, 68, 331, 225, 71, 375, 580, 64, 607, 616, 124,
            712, 107, 270, 120, 681, 185, 391, 499, 122, 47, 569, 418, 452, 676, 614, 476,
            29, 651, 61, 527, 601, 691, 226, 407, 382, 314, 128, 726, 357, 217, 641, 680,
            386, 4, 36, 178, 643, 263, 5, 757, 228, 313, 261, 241, 208, 652, 683, 142,
            284, 340, 216, 44, 65, 461, 484, 345, 51, 433, 449, 581, 110, 425, 94, 454,
            568, 649, 282, 78, 104, 657, 338, 38, 689, 376, 505, 354, 388, 175, 474, 509,
            330, 176, 207, 558, 323, 230, 740, 500, 519, 31, 711, 126, 146, 579, 220, 300,
            39, 306, 654, 512, 703, 399, 459, 684, 548, 309, 566, 406, 502, 687, 412, 619,
            450, 43, 624, 96, 136, 105, 694, 759, 58, 576, 620, 705, 312, 667, 112, 604,
            6, 561, 231, 536, 150, 13, 184, 169, 458, 697, 572, 575, 147, 75, 321, 380,
            695, 111, 298, 599, 373, 494, 556, 730, 729, 545, 611, 34, 276, 408, 610, 257,
            715, 42, 636, 584, 710, 632, 410, 114, 665, 49, 197, 265, 366, 592, 90, 560,
            562, 342, 685, 693, 506, 585, 612, 436, 204, 262, 764, 189, 73, 269, 87, 672,
            292, 434, 557, 663, 195, 404, 88, 367, 671, 74, 421, 530, 627, 360, 273, 15,
            396, 567, 324, 749, 574, 448, 356, 246, 315, 732, 754, 485, 129, 9, 488, 102,
            134, 319, 658, 721, 442, 758, 362, 132, 53, 543, 140, 628, 256, 287, 77, 595,
            463, 45, 745, 268, 30, 8, 188, 559, 647, 213, 389, 668, 716, 645, 158, 343,
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

        extern std::array<u64, L1_SIZE> NNZCounts;
        extern u64 ActivationCount;
        extern u64 EvalCalls;
    }

}

#endif
