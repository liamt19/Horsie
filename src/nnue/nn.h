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
            174, 266, 369, 334, 109, 501, 696, 742, 125, 299, 622, 173, 238, 130, 355, 727,
            475, 341, 429, 535, 756, 252, 479, 508, 457, 326, 33, 285, 385, 766, 123, 415,
            609, 251, 547, 244, 359, 405, 218, 19, 143, 481, 507, 0, 718, 336, 172, 332,
            259, 427, 118, 596, 762, 430, 482, 374, 163, 692, 496, 414, 70, 24, 305, 211,
            32, 152, 181, 724, 570, 63, 91, 364, 542, 555, 322, 79, 642, 446, 439, 347,
            160, 707, 445, 597, 395, 316, 131, 295, 603, 714, 578, 504, 673, 660, 422, 765,
            416, 371, 2, 155, 497, 187, 532, 48, 223, 605, 279, 602, 198, 135, 526, 471,
            46, 503, 203, 639, 186, 565, 688, 3, 350, 249, 191, 92, 443, 571, 62, 761,
            394, 199, 237, 401, 275, 625, 648, 202, 137, 365, 333, 138, 304, 438, 358, 57,
            465, 467, 702, 368, 588, 286, 1, 608, 182, 477, 466, 733, 86, 554, 413, 720,
            212, 460, 630, 661, 363, 490, 151, 582, 682, 613, 278, 637, 294, 666, 487, 573,
            97, 215, 264, 272, 686, 480, 329, 586, 267, 227, 153, 221, 737, 447, 115, 538,
            650, 551, 731, 168, 210, 468, 708, 28, 495, 377, 296, 719, 121, 240, 618, 659,
            166, 352, 589, 469, 709, 744, 690, 116, 735, 192, 522, 247, 669, 310, 260, 489,
            440, 99, 486, 353, 119, 653, 233, 462, 679, 428, 209, 411, 600, 594, 472, 235,
            101, 117, 320, 165, 546, 674, 739, 59, 514, 524, 748, 81, 423, 206, 767, 392,
            18, 432, 361, 587, 348, 760, 274, 728, 409, 662, 14, 243, 590, 239, 398, 127,
            236, 528, 640, 337, 83, 723, 577, 498, 302, 20, 67, 72, 229, 280, 93, 317,
            100, 644, 473, 736, 583, 283, 339, 402, 431, 491, 148, 250, 550, 50, 76, 325,
            200, 540, 205, 346, 16, 451, 699, 722, 35, 752, 537, 349, 288, 141, 493, 529,
            378, 224, 351, 515, 21, 403, 60, 95, 85, 553, 520, 54, 725, 563, 615, 255,
            258, 633, 248, 419, 271, 26, 66, 157, 113, 10, 98, 397, 539, 437, 606, 455,
            635, 678, 701, 393, 646, 41, 303, 717, 670, 511, 84, 426, 534, 492, 634, 591,
            381, 89, 162, 214, 139, 677, 747, 521, 638, 372, 763, 564, 106, 82, 750, 593,
            598, 741, 483, 738, 183, 25, 734, 254, 478, 170, 171, 156, 194, 190, 253, 331,
            370, 531, 523, 308, 706, 69, 234, 656, 518, 193, 743, 675, 525, 424, 700, 541,
            453, 420, 513, 390, 327, 318, 230, 441, 470, 400, 290, 549, 753, 301, 384, 631,
            516, 80, 180, 7, 544, 704, 291, 713, 245, 159, 464, 56, 435, 201, 311, 552,
            456, 383, 219, 751, 145, 196, 307, 289, 133, 517, 510, 103, 222, 23, 161, 655,
            746, 297, 22, 71, 629, 621, 328, 51, 626, 167, 375, 37, 242, 232, 293, 387,
            379, 712, 277, 17, 144, 12, 664, 27, 691, 108, 623, 344, 533, 52, 149, 178,
            313, 61, 281, 617, 616, 338, 68, 40, 44, 47, 569, 185, 314, 241, 527, 755,
            452, 476, 417, 726, 55, 614, 418, 270, 444, 580, 683, 335, 757, 64, 226, 65,
            179, 607, 107, 354, 154, 225, 652, 122, 680, 499, 263, 120, 29, 676, 177, 261,
            357, 711, 641, 142, 698, 651, 208, 228, 425, 11, 391, 643, 340, 407, 124, 657,
            128, 216, 382, 433, 548, 38, 568, 282, 681, 558, 5, 330, 112, 345, 94, 601,
            104, 386, 512, 4, 217, 164, 461, 649, 581, 126, 306, 502, 284, 459, 519, 449,
            146, 689, 654, 78, 36, 505, 39, 388, 454, 484, 13, 406, 412, 376, 110, 136,
            619, 759, 175, 31, 43, 509, 500, 740, 207, 566, 220, 323, 309, 561, 576, 703,
            399, 579, 96, 450, 604, 184, 729, 730, 58, 687, 321, 408, 684, 620, 575, 705,
            474, 105, 147, 300, 458, 169, 693, 624, 611, 231, 599, 697, 612, 176, 150, 312,
            49, 536, 298, 572, 632, 6, 556, 694, 373, 685, 42, 75, 265, 380, 410, 195,
            715, 585, 257, 665, 114, 710, 545, 636, 494, 695, 667, 610, 111, 204, 562, 366,
            764, 584, 197, 189, 506, 342, 73, 560, 292, 262, 592, 276, 360, 256, 436, 88,
            672, 34, 269, 663, 567, 90, 671, 434, 404, 74, 749, 627, 543, 367, 530, 421,
            754, 396, 721, 485, 246, 140, 273, 557, 129, 9, 324, 134, 87, 356, 102, 30,
            732, 319, 15, 448, 574, 132, 745, 315, 758, 362, 268, 53, 213, 488, 628, 463,
            658, 8, 188, 77, 442, 287, 647, 595, 45, 559, 389, 668, 716, 645, 158, 343,
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
