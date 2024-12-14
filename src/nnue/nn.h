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

        bool IsCompressed(std::istream& stream);
        void LoadZSTD(std::istream& m_stream, std::byte* dst);
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


        constexpr i32 BestPermuteIndices[] = {
            374, 365, 344, 703, 624, 362, 677, 462, 485, 598, 69, 455, 457, 241, 147, 755,
            666, 55, 712, 315, 164, 146, 520, 551, 224, 181, 182, 49, 177, 227, 186, 306,
            746, 307, 754, 115, 410, 135, 157, 179, 433, 50, 126, 423, 707, 333, 560, 425,
            726, 639, 709, 739, 70, 638, 122, 478, 353, 154, 404, 631, 453, 349, 328, 367,
            208, 698, 72, 95, 759, 736, 165, 539, 254, 390, 535, 643, 3, 57, 23, 719,
            456, 281, 583, 336, 386, 441, 671, 139, 200, 557, 150, 347, 401, 573, 14, 133,
            732, 235, 584, 437, 411, 621, 767, 346, 35, 41, 368, 745, 85, 103, 679, 369,
            84, 757, 403, 34, 360, 608, 299, 1, 345, 519, 275, 117, 18, 764, 570, 185,
            645, 293, 689, 578, 443, 634, 657, 375, 132, 76, 710, 77, 121, 451, 364, 357,
            594, 144, 282, 680, 599, 119, 108, 111, 334, 503, 290, 225, 371, 397, 445, 296,
            244, 735, 399, 724, 337, 22, 289, 11, 381, 715, 264, 329, 107, 459, 765, 252,
            172, 515, 189, 574, 168, 420, 651, 730, 506, 156, 713, 742, 291, 589, 571, 641,
            195, 636, 320, 575, 513, 607, 623, 387, 545, 234, 267, 319, 120, 113, 613, 392,
            47, 268, 454, 555, 714, 199, 188, 247, 426, 534, 13, 616, 64, 263, 285, 469,
            348, 685, 619, 114, 86, 272, 276, 614, 123, 734, 667, 491, 124, 297, 460, 400,
            87, 465, 554, 488, 688, 738, 298, 20, 522, 309, 63, 217, 533, 174, 262, 648,
            563, 332, 221, 721, 558, 434, 245, 338, 466, 611, 383, 16, 373, 741, 763, 492,
            260, 45, 286, 193, 436, 502, 343, 603, 729, 516, 517, 342, 694, 158, 237, 206,
            699, 238, 438, 194, 2, 153, 109, 637, 243, 138, 622, 251, 702, 670, 597, 242,
            541, 38, 21, 484, 618, 416, 192, 300, 674, 56, 722, 42, 127, 683, 82, 505,
            173, 202, 626, 233, 145, 96, 9, 565, 590, 257, 30, 151, 660, 314, 617, 112,
            183, 89, 544, 446, 561, 733, 316, 384, 656, 629, 602, 99, 431, 716, 212, 669,
            760, 131, 542, 620, 17, 355, 407, 305, 283, 83, 270, 134, 633, 751, 615, 532,
            28, 304, 725, 552, 169, 140, 498, 600, 672, 219, 531, 676, 612, 363, 748, 52,
            0, 701, 470, 180, 644, 744, 7, 483, 650, 81, 606, 60, 58, 447, 458, 59,
            523, 71, 653, 627, 700, 536, 91, 749, 418, 605, 68, 40, 75, 354, 429, 585,
            116, 175, 226, 46, 463, 461, 537, 409, 546, 682, 118, 711, 432, 704, 582, 538,
            48, 295, 214, 277, 642, 178, 385, 287, 419, 228, 253, 486, 592, 61, 635, 652,
            452, 136, 159, 210, 662, 43, 8, 530, 692, 32, 405, 162, 294, 73, 628, 161,
            325, 327, 160, 731, 382, 155, 693, 143, 664, 377, 356, 580, 62, 190, 4, 292,
            402, 473, 176, 511, 339, 53, 412, 366, 303, 65, 567, 475, 250, 504, 675, 435,
            524, 528, 691, 665, 415, 587, 549, 690, 678, 74, 19, 439, 51, 106, 330, 658,
            29, 335, 25, 471, 509, 205, 273, 98, 141, 540, 512, 442, 246, 647, 93, 321,
            197, 5, 490, 427, 547, 548, 129, 632, 340, 211, 184, 588, 525, 663, 581, 720,
            681, 708, 288, 198, 586, 220, 279, 239, 256, 526, 496, 223, 489, 24, 54, 231,
            36, 750, 311, 495, 33, 477, 148, 654, 727, 421, 450, 449, 318, 302, 556, 481,
            6, 543, 514, 661, 170, 743, 476, 388, 723, 576, 350, 236, 468, 494, 232, 408,
            507, 201, 752, 372, 406, 310, 130, 577, 380, 213, 649, 97, 313, 609, 376, 94,
            216, 324, 191, 265, 758, 424, 125, 550, 497, 646, 163, 728, 308, 10, 12, 27,
            596, 591, 668, 527, 687, 753, 105, 398, 579, 149, 312, 249, 640, 15, 240, 167,
            705, 480, 740, 44, 472, 493, 110, 222, 171, 215, 625, 413, 37, 518, 747, 391,
            209, 610, 90, 417, 601, 152, 67, 686, 322, 326, 261, 166, 508, 301, 428, 31,
            695, 479, 187, 448, 230, 317, 258, 559, 207, 562, 566, 26, 655, 553, 673, 467,
            766, 568, 341, 762, 684, 88, 717, 510, 66, 396, 521, 358, 501, 379, 352, 474,
            630, 128, 378, 78, 659, 274, 394, 142, 564, 422, 80, 370, 280, 595, 331, 79,
            137, 204, 278, 487, 414, 706, 101, 361, 323, 284, 464, 572, 395, 569, 430, 102,
            104, 351, 203, 696, 529, 499, 218, 255, 756, 761, 482, 248, 92, 440, 697, 196,
            266, 718, 259, 269, 389, 100, 359, 737, 500, 229, 604, 39, 271, 393, 593, 444,
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
