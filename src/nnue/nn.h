#pragma once

#define PERM_COUNT 1
#undef PERM_COUNT

#define NO_PERM 1
#undef NO_PERM

#include "../defs.h"
#include "../nnue/arch.h"
#include "../position.h"
#include "accumulator.h"

#include <array>
#include <span>

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
            U L2SWeights[L2_SKIP_SIZE][OUTPUT_BUCKETS][L3_HALF_SIZE];
            U L2SBiases[OUTPUT_BUCKETS][L3_HALF_SIZE];
            U L2CWeights[L2_SKIP_SIZE][OUTPUT_BUCKETS][L3_HALF_SIZE];
            U L2CBiases[OUTPUT_BUCKETS][L3_HALF_SIZE];
            U L3Weights[L3_SIZE][OUTPUT_BUCKETS];
            U L3Biases[OUTPUT_BUCKETS];
        };
        using QuantisedNetwork = QuantisedNetworkBase<i16, i8, float>;

        template <typename T, typename W, typename U>
        struct alignas(64) NetworkBase {
            std::array<T, INPUT_SIZE * L1_SIZE * INPUT_BUCKETS>                   FTWeights;
            std::array<T, L1_SIZE>                                                FTBiases;
            std::array<std::array<W, L1_SIZE * L2_SIZE         >, OUTPUT_BUCKETS> L1Weights;
            std::array<std::array<U, L2_SIZE                   >, OUTPUT_BUCKETS> L1Biases;
            std::array<std::array<U, L2_SKIP_SIZE* L3_HALF_SIZE>, OUTPUT_BUCKETS> L2SWeights;
            std::array<std::array<U, L3_HALF_SIZE              >, OUTPUT_BUCKETS> L2SBiases;
            std::array<std::array<U, L2_SKIP_SIZE* L3_HALF_SIZE>, OUTPUT_BUCKETS> L2CWeights;
            std::array<std::array<U, L3_HALF_SIZE              >, OUTPUT_BUCKETS> L2CBiases;
            std::array<std::array<U, L3_SIZE                   >, OUTPUT_BUCKETS> L3Weights;
            std::array<U, OUTPUT_BUCKETS>                                         L3Biases;
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


        std::pair<i32, i32> FeatureIndex(i32 pc, i32 pt, i32 sq, i32 wk, i32 bk);
        constexpr i32 FeatureIndexSingle(i32 pc, i32 pt, i32 sq, i32 kingSq, i32 perspective);

        constexpr i32 BucketScheme[] = {
            0, 1, 2, 3,
            0, 1, 2, 3,
            4, 4, 5, 5,
            4, 4, 5, 5,
            6, 6, 7, 7,
            6, 6, 7, 7,
            8, 8, 9, 9,
            8, 8, 9, 9,
        };

        constexpr auto KingBuckets = []
        {
            std::array<i32, 64> arr{};

            constexpr i32 MirroredIndices[] = { 0, 1, 2, 3, 3, 2, 1, 0 };
            const auto N_BUCKETS = 1 + *std::max_element(BucketScheme, BucketScheme + (sizeof(BucketScheme) / sizeof(i32)));
            
            for (i32 y = 0; y < 8; y++)
                for (i32 x = 0; x < 8; x++)
                    arr[CoordToIndex(x, y)] = BucketScheme[(y * 4) + MirroredIndices[x]] + (x > 3 ? N_BUCKETS : 0);

            return arr;
        }();

        constexpr i32 Orient(i32 sq, i32 kingSq, i32 perspective) {
            return (sq ^ (56 * perspective) ^ (kingSq % 8 > 3 ? 7 : 0));
        }

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
            95, 706, 223, 190, 256, 386, 168, 194, 18, 139, 399, 89, 411, 531, 345, 272,
            596, 243, 658, 94, 492, 108, 554, 141, 129, 709, 648, 325, 393, 21, 574, 741,
            211, 109, 175, 729, 381, 465, 9, 748, 312, 47, 518, 625, 713, 73, 476, 526,
            577, 50, 384, 629, 282, 764, 410, 549, 275, 679, 274, 496, 320, 693, 292, 705,
            102, 179, 443, 572, 517, 252, 377, 313, 3, 146, 497, 26, 631, 544, 119, 481,
            747, 17, 215, 570, 75, 153, 647, 364, 663, 449, 212, 585, 25, 701, 684, 122,
            124, 308, 676, 598, 302, 301, 121, 360, 368, 306, 185, 337, 374, 394, 633, 731,
            662, 49, 426, 208, 88, 507, 191, 548, 644, 249, 442, 79, 495, 593, 241, 171,
            480, 626, 652, 342, 4, 762, 700, 716, 397, 314, 436, 385, 685, 688, 245, 754,
            182, 232, 565, 151, 563, 367, 254, 82, 512, 58, 80, 656, 349, 438, 174, 678,
            712, 650, 402, 763, 372, 164, 157, 138, 125, 46, 392, 86, 692, 323, 454, 96,
            649, 638, 404, 136, 355, 379, 660, 33, 687, 623, 541, 328, 99, 418, 669, 621,
            588, 489, 336, 100, 172, 32, 704, 550, 27, 39, 616, 620, 203, 10, 356, 178,
            659, 396, 339, 445, 720, 369, 206, 247, 74, 388, 425, 307, 42, 145, 641, 758,
            472, 714, 439, 259, 59, 567, 619, 165, 630, 309, 344, 589, 116, 603, 300, 285,
            55, 573, 677, 734, 757, 750, 239, 280, 539, 63, 538, 83, 753, 112, 681, 605,
            462, 107, 584, 214, 321, 354, 732, 617, 501, 192, 578, 218, 267, 376, 16, 117,
            600, 599, 93, 542, 450, 8, 199, 120, 503, 135, 461, 407, 195, 71, 675, 423,
            521, 433, 68, 702, 276, 204, 148, 193, 216, 298, 711, 357, 766, 628, 281, 475,
            238, 207, 583, 726, 640, 467, 161, 459, 519, 30, 382, 533, 636, 84, 661, 722,
            571, 64, 365, 703, 375, 416, 149, 463, 756, 2, 604, 187, 233, 315, 110, 155,
            494, 378, 566, 294, 167, 710, 283, 651, 40, 131, 160, 304, 689, 665, 130, 422,
            391, 730, 485, 430, 261, 293, 269, 333, 104, 105, 728, 44, 455, 724, 268, 210,
            499, 760, 0, 322, 311, 613, 359, 511, 228, 637, 184, 514, 525, 483, 248, 668,
            646, 37, 579, 91, 326, 227, 546, 253, 35, 458, 708, 251, 373, 403, 358, 159,
            147, 643, 466, 390, 683, 673, 695, 664, 727, 653, 552, 742, 350, 420, 740, 725,
            324, 106, 632, 200, 115, 20, 428, 170, 297, 81, 733, 69, 672, 470, 142, 471,
            226, 406, 242, 590, 118, 166, 28, 132, 383, 123, 13, 152, 288, 767, 614, 674,
            473, 158, 224, 715, 101, 556, 505, 736, 510, 412, 635, 610, 657, 749, 424, 90,
            362, 543, 45, 338, 487, 370, 515, 284, 413, 97, 612, 765, 346, 509, 271, 535,
            380, 162, 513, 721, 341, 387, 686, 5, 201, 144, 432, 671, 735, 417, 738, 290,
            143, 54, 670, 67, 230, 490, 289, 624, 202, 263, 609, 331, 606, 759, 237, 645,
            654, 85, 29, 43, 257, 448, 353, 48, 622, 437, 246, 634, 427, 60, 278, 655,
            478, 582, 327, 395, 134, 719, 221, 156, 491, 615, 277, 205, 41, 343, 113, 751,
            103, 72, 562, 504, 303, 666, 694, 222, 524, 457, 258, 270, 329, 560, 197, 468,
            627, 76, 555, 537, 352, 266, 183, 482, 173, 22, 444, 486, 389, 6, 516, 469,
            421, 398, 484, 409, 601, 196, 264, 348, 569, 24, 575, 51, 286, 401, 98, 576,
            500, 57, 551, 255, 128, 520, 557, 488, 698, 618, 176, 739, 498, 319, 452, 244,
            114, 189, 587, 707, 682, 273, 291, 696, 56, 62, 316, 745, 479, 351, 667, 608,
            127, 534, 77, 234, 31, 169, 181, 361, 92, 419, 87, 414, 558, 310, 460, 52,
            363, 508, 126, 262, 586, 527, 1, 405, 446, 12, 415, 7, 547, 140, 761, 737,
            260, 305, 744, 580, 65, 532, 477, 464, 528, 435, 330, 14, 553, 66, 335, 209,
            34, 493, 502, 15, 177, 296, 295, 718, 366, 746, 231, 559, 691, 225, 23, 755,
            602, 529, 70, 299, 265, 340, 150, 530, 717, 540, 523, 11, 536, 229, 639, 440,
            408, 163, 611, 642, 690, 198, 752, 680, 317, 434, 561, 451, 332, 581, 78, 592,
            347, 723, 287, 240, 453, 36, 607, 38, 19, 545, 400, 235, 250, 111, 217, 371,
            213, 506, 154, 220, 236, 334, 53, 456, 447, 61, 591, 180, 431, 219, 595, 743,
            697, 137, 188, 318, 133, 279, 597, 594, 564, 474, 568, 186, 699, 429, 441, 522,
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
