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
            374, 344, 365, 703, 624, 677, 362, 485, 462, 598, 69, 455, 457, 241, 755, 666,
            147, 55, 146, 712, 315, 164, 551, 181, 520, 49, 177, 182, 224, 410, 227, 186,
            115, 135, 306, 754, 307, 746, 50, 333, 126, 739, 179, 433, 157, 425, 726, 639,
            707, 478, 560, 423, 122, 453, 349, 70, 404, 736, 154, 709, 367, 638, 759, 631,
            254, 353, 95, 208, 165, 390, 328, 72, 539, 583, 671, 456, 281, 23, 535, 698,
            103, 719, 643, 401, 584, 767, 573, 139, 386, 3, 411, 150, 346, 441, 557, 732,
            57, 336, 85, 14, 133, 41, 235, 347, 84, 679, 745, 621, 403, 375, 437, 369,
            608, 18, 34, 519, 86, 35, 368, 299, 200, 757, 108, 345, 443, 275, 119, 360,
            710, 1, 570, 185, 117, 121, 578, 764, 76, 77, 132, 689, 680, 111, 599, 634,
            371, 397, 144, 293, 282, 645, 451, 594, 296, 381, 445, 290, 11, 715, 337, 765,
            289, 735, 168, 420, 329, 22, 364, 459, 503, 334, 657, 156, 225, 357, 730, 172,
            244, 515, 264, 291, 742, 607, 651, 399, 724, 506, 252, 636, 107, 713, 426, 589,
            195, 268, 571, 387, 113, 574, 545, 13, 124, 188, 614, 267, 513, 641, 320, 247,
            623, 120, 189, 234, 263, 319, 199, 575, 734, 590, 613, 469, 454, 714, 616, 555,
            298, 465, 554, 534, 721, 20, 667, 297, 611, 272, 460, 685, 688, 392, 738, 491,
            619, 64, 383, 522, 338, 348, 87, 123, 285, 63, 47, 466, 488, 563, 217, 648,
            343, 45, 332, 694, 492, 286, 245, 114, 502, 558, 221, 373, 400, 16, 309, 109,
            174, 238, 173, 276, 436, 434, 763, 517, 262, 637, 193, 626, 741, 2, 158, 533,
            42, 145, 597, 627, 565, 699, 202, 438, 260, 702, 722, 729, 206, 342, 194, 683,
            21, 603, 622, 484, 82, 670, 242, 30, 516, 192, 237, 733, 127, 233, 17, 183,
            138, 541, 716, 316, 561, 300, 153, 620, 257, 89, 416, 617, 618, 751, 674, 314,
            56, 602, 96, 99, 9, 243, 629, 407, 151, 676, 251, 760, 112, 446, 384, 544,
            253, 270, 470, 355, 505, 546, 542, 660, 656, 81, 169, 531, 725, 131, 435, 363,
            134, 552, 283, 536, 633, 672, 748, 305, 91, 8, 140, 304, 38, 212, 650, 75,
            28, 498, 642, 7, 53, 40, 219, 68, 178, 52, 458, 701, 48, 60, 180, 606,
            669, 354, 431, 432, 605, 483, 644, 116, 700, 749, 532, 744, 600, 615, 429, 46,
            83, 0, 538, 704, 612, 71, 175, 447, 523, 277, 58, 592, 463, 653, 136, 419,
            452, 652, 226, 93, 418, 385, 214, 295, 635, 693, 325, 228, 162, 662, 530, 59,
            118, 682, 664, 65, 528, 582, 292, 711, 356, 294, 190, 327, 73, 339, 409, 628,
            61, 159, 303, 475, 377, 540, 524, 537, 580, 155, 473, 511, 161, 471, 461, 160,
            129, 665, 143, 690, 442, 486, 106, 412, 51, 287, 19, 43, 335, 490, 658, 512,
            405, 246, 32, 198, 211, 585, 439, 586, 691, 402, 731, 675, 720, 62, 567, 382,
            692, 250, 4, 176, 366, 504, 340, 587, 547, 25, 549, 273, 74, 141, 311, 663,
            29, 330, 509, 588, 477, 210, 220, 98, 581, 5, 678, 427, 279, 647, 548, 197,
            576, 205, 33, 415, 321, 526, 148, 36, 256, 239, 223, 495, 750, 54, 231, 591,
            727, 489, 494, 525, 543, 184, 302, 654, 632, 421, 378, 681, 514, 408, 449, 649,
            496, 388, 6, 507, 380, 201, 170, 24, 213, 708, 723, 481, 97, 450, 324, 577,
            310, 130, 752, 668, 468, 646, 372, 288, 94, 236, 424, 216, 609, 556, 265, 376,
            743, 497, 313, 318, 44, 350, 232, 527, 105, 740, 480, 240, 661, 301, 476, 125,
            10, 596, 152, 508, 167, 308, 728, 149, 758, 191, 27, 753, 312, 37, 550, 406,
            163, 398, 171, 687, 222, 230, 209, 322, 261, 67, 249, 12, 579, 705, 326, 15,
            625, 166, 640, 479, 215, 562, 317, 686, 610, 417, 358, 601, 90, 747, 566, 518,
            472, 187, 31, 493, 391, 467, 110, 79, 428, 717, 559, 655, 448, 762, 413, 766,
            78, 673, 142, 695, 568, 258, 341, 88, 207, 26, 553, 564, 474, 684, 501, 379,
            510, 66, 396, 422, 521, 352, 80, 487, 595, 274, 630, 128, 101, 137, 204, 464,
            280, 659, 331, 414, 370, 394, 278, 361, 284, 395, 104, 706, 529, 572, 218, 569,
            430, 323, 203, 102, 696, 499, 351, 92, 756, 482, 255, 697, 248, 761, 196, 440,
            259, 718, 269, 500, 266, 359, 389, 737, 100, 271, 393, 229, 604, 39, 593, 444,
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
