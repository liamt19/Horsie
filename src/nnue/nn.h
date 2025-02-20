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
            U L2Weights[L2_SIZE][OUTPUT_BUCKETS][L3_SIZE];
            U L2Biases[OUTPUT_BUCKETS][L3_SIZE];
            U L3Weights[L3_SIZE][OUTPUT_BUCKETS];
            U L3Biases[OUTPUT_BUCKETS];
        };
        using QuantisedNetwork = QuantisedNetworkBase<i16, i8, float>;

        template <typename T, typename W, typename U>
        struct alignas(64) NetworkBase {
            std::array<T, INPUT_SIZE * L1_SIZE * INPUT_BUCKETS>          FTWeights;
            std::array<T, L1_SIZE>                                       FTBiases;
            std::array<std::array<W, L1_SIZE * L2_SIZE>, OUTPUT_BUCKETS> L1Weights;
            std::array<std::array<U, L2_SIZE          >, OUTPUT_BUCKETS> L1Biases;
            std::array<std::array<U, L2_SIZE * L3_SIZE>, OUTPUT_BUCKETS> L2Weights;
            std::array<std::array<U, L3_SIZE          >, OUTPUT_BUCKETS> L2Biases;
            std::array<std::array<U, L3_SIZE>,           OUTPUT_BUCKETS> L3Weights;
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
            446, 543, 690, 614, 96, 705, 123, 544, 79, 225, 472, 234, 333, 732, 721, 34,
            720, 279, 433, 360, 512, 431, 714, 50, 764, 566, 252, 231, 466, 348, 16, 58,
            453, 240, 585, 624, 452, 78, 610, 503, 166, 29, 292, 650, 285, 524, 683, 189,
            60, 531, 643, 522, 754, 99, 138, 722, 687, 488, 498, 97, 478, 41, 31, 144,
            601, 271, 101, 500, 351, 668, 475, 424, 266, 140, 528, 313, 173, 550, 9, 94,
            26, 589, 655, 143, 508, 335, 712, 642, 342, 657, 256, 131, 742, 145, 378, 449,
            339, 455, 672, 459, 562, 341, 62, 192, 168, 42, 2, 658, 724, 602, 198, 320,
            322, 671, 308, 619, 129, 400, 421, 404, 598, 98, 553, 501, 440, 695, 398, 396,
            307, 353, 489, 713, 17, 560, 637, 346, 753, 49, 86, 128, 213, 376, 401, 10,
            457, 664, 676, 718, 349, 182, 174, 305, 276, 509, 411, 644, 651, 238, 580, 70,
            592, 423, 212, 723, 386, 747, 215, 587, 319, 82, 375, 390, 412, 328, 204, 81,
            518, 413, 132, 121, 247, 640, 693, 436, 533, 68, 237, 372, 545, 387, 289, 115,
            24, 547, 118, 302, 11, 707, 652, 370, 340, 706, 519, 418, 180, 104, 597, 385,
            211, 171, 620, 689, 698, 114, 107, 35, 155, 156, 606, 361, 136, 666, 382, 229,
            593, 634, 334, 611, 410, 205, 388, 39, 546, 254, 273, 184, 648, 752, 755, 27,
            133, 36, 324, 306, 590, 203, 23, 407, 291, 748, 639, 631, 419, 437, 645, 568,
            744, 126, 150, 438, 495, 354, 760, 286, 259, 463, 746, 142, 164, 520, 21, 447,
            248, 617, 196, 600, 159, 296, 343, 102, 329, 91, 487, 278, 73, 325, 241, 445,
            719, 766, 479, 176, 350, 55, 226, 112, 534, 218, 44, 504, 202, 573, 420, 563,
            374, 667, 134, 119, 54, 702, 523, 258, 275, 565, 207, 661, 32, 83, 660, 250,
            337, 728, 394, 653, 22, 152, 406, 582, 623, 312, 456, 100, 175, 85, 441, 586,
            647, 257, 347, 514, 700, 527, 239, 704, 37, 510, 183, 4, 141, 558, 357, 638,
            178, 435, 224, 263, 625, 135, 331, 52, 53, 124, 681, 575, 116, 450, 179, 596,
            468, 511, 629, 571, 255, 761, 206, 677, 157, 633, 294, 684, 578, 246, 481, 448,
            244, 735, 542, 8, 57, 579, 187, 230, 646, 572, 197, 236, 280, 758, 109, 480,
            709, 581, 14, 493, 434, 111, 402, 300, 745, 539, 464, 506, 694, 304, 303, 249,
            670, 430, 366, 627, 232, 686, 680, 262, 356, 496, 605, 570, 122, 584, 181, 669,
            485, 540, 460, 380, 186, 507, 505, 701, 67, 541, 1, 13, 274, 105, 264, 403,
            473, 490, 417, 515, 235, 685, 311, 298, 594, 219, 710, 733, 383, 220, 108, 137,
            5, 554, 283, 154, 729, 270, 692, 120, 526, 199, 461, 635, 529, 317, 548, 751,
            675, 125, 69, 194, 696, 626, 759, 663, 7, 555, 494, 591, 217, 731, 323, 209,
            327, 697, 439, 46, 19, 282, 654, 622, 214, 636, 530, 149, 756, 316, 147, 603,
            288, 604, 190, 251, 165, 210, 576, 191, 384, 615, 662, 309, 716, 736, 444, 465,
            462, 92, 268, 426, 77, 51, 15, 725, 269, 414, 577, 93, 158, 659, 167, 682,
            476, 284, 12, 405, 688, 613, 84, 535, 153, 330, 72, 193, 583, 332, 245, 30,
            61, 730, 422, 208, 656, 290, 567, 492, 458, 491, 556, 76, 45, 483, 628, 95,
            0, 318, 40, 74, 484, 315, 393, 272, 368, 170, 749, 71, 649, 242, 416, 399,
            222, 467, 59, 43, 160, 28, 261, 87, 358, 103, 608, 227, 88, 139, 739, 295,
            195, 47, 243, 762, 299, 365, 641, 265, 679, 442, 287, 757, 18, 486, 717, 146,
            517, 451, 537, 738, 163, 425, 691, 355, 151, 678, 3, 551, 106, 632, 169, 373,
            364, 750, 432, 33, 379, 321, 277, 281, 607, 427, 75, 552, 177, 389, 740, 765,
            310, 260, 538, 20, 665, 200, 216, 595, 763, 588, 569, 499, 474, 708, 502, 532,
            297, 117, 741, 397, 233, 223, 561, 477, 6, 470, 482, 516, 767, 38, 89, 395,
            359, 454, 557, 113, 267, 90, 363, 408, 381, 326, 443, 525, 161, 521, 130, 392,
            743, 703, 336, 536, 699, 110, 253, 574, 221, 162, 559, 711, 228, 564, 715, 513,
            674, 80, 201, 673, 25, 127, 726, 727, 599, 377, 497, 391, 429, 185, 734, 371,
            148, 737, 369, 367, 428, 409, 66, 630, 65, 471, 64, 63, 609, 172, 338, 469,
            612, 301, 56, 344, 362, 616, 188, 618, 48, 345, 314, 415, 621, 352, 549, 293,
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
