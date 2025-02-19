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
            U L2SWeights[L2_SIZE][OUTPUT_BUCKETS][L3_HALF_SIZE];
            U L2SBiases[OUTPUT_BUCKETS][L3_HALF_SIZE];
            U L2CWeights[L2_SIZE][OUTPUT_BUCKETS][L3_HALF_SIZE];
            U L2CBiases[OUTPUT_BUCKETS][L3_HALF_SIZE];
            U L3Weights[L3_SIZE][OUTPUT_BUCKETS];
            U L3Biases[OUTPUT_BUCKETS];
        };
        using QuantisedNetwork = QuantisedNetworkBase<i16, i8, float>;

        template <typename T, typename W, typename U>
        struct alignas(64) NetworkBase {
            std::array<T, INPUT_SIZE * L1_SIZE * INPUT_BUCKETS>               FTWeights;
            std::array<T, L1_SIZE>                                            FTBiases;
            std::array<std::array<W, L1_SIZE * L2_SIZE>,      OUTPUT_BUCKETS> L1Weights;
            std::array<std::array<U, L2_SIZE          >,      OUTPUT_BUCKETS> L1Biases;
            std::array<std::array<U, L2_SIZE * L3_HALF_SIZE>, OUTPUT_BUCKETS> L2SWeights;
            std::array<std::array<U, L3_HALF_SIZE          >, OUTPUT_BUCKETS> L2SBiases;
            std::array<std::array<U, L2_SIZE * L3_HALF_SIZE>, OUTPUT_BUCKETS> L2CWeights;
            std::array<std::array<U, L3_HALF_SIZE          >, OUTPUT_BUCKETS> L2CBiases;
            std::array<std::array<U, L3_SIZE               >, OUTPUT_BUCKETS> L3Weights;
            std::array<U, OUTPUT_BUCKETS>                                     L3Biases;
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
            47, 324, 114, 754, 394, 59, 442, 481, 159, 219, 585, 611, 512, 534, 127, 537,
            759, 229, 261, 201, 586, 372, 228, 225, 400, 522, 406, 368, 678, 598, 7, 469,
            240, 328, 135, 574, 206, 152, 177, 423, 254, 30, 200, 332, 262, 413, 121, 89,
            725, 704, 578, 326, 168, 433, 660, 756, 39, 432, 347, 305, 81, 401, 497, 440,
            202, 97, 140, 728, 523, 0, 583, 255, 155, 397, 26, 253, 532, 196, 3, 531,
            629, 227, 370, 490, 125, 393, 43, 204, 736, 548, 132, 471, 567, 2, 577, 44,
            407, 158, 21, 618, 191, 632, 55, 726, 627, 147, 464, 721, 596, 513, 446, 753,
            171, 283, 664, 436, 733, 514, 334, 623, 591, 375, 699, 416, 236, 279, 554, 188,
            474, 417, 364, 101, 226, 589, 335, 96, 80, 740, 157, 187, 194, 675, 88, 562,
            195, 346, 448, 749, 659, 669, 405, 338, 657, 286, 470, 670, 386, 51, 727, 505,
            367, 547, 600, 58, 320, 424, 662, 686, 77, 706, 98, 410, 682, 224, 9, 15,
            377, 31, 248, 507, 418, 650, 267, 29, 62, 165, 296, 717, 82, 360, 256, 665,
            685, 217, 472, 766, 679, 100, 316, 330, 508, 612, 765, 348, 677, 732, 668, 336,
            571, 302, 309, 294, 198, 530, 609, 494, 34, 143, 349, 592, 209, 720, 687, 323,
            90, 655, 376, 533, 345, 223, 144, 714, 71, 173, 91, 558, 149, 298, 1, 109,
            129, 724, 172, 221, 391, 351, 434, 231, 210, 331, 502, 87, 104, 280, 742, 272,
            630, 174, 739, 468, 333, 460, 661, 560, 748, 651, 161, 362, 124, 689, 549, 14,
            322, 511, 480, 631, 716, 518, 19, 539, 403, 607, 603, 646, 457, 69, 667, 184,
            85, 291, 605, 341, 297, 746, 220, 266, 359, 16, 354, 41, 559, 622, 613, 137,
            690, 526, 499, 249, 702, 42, 142, 399, 529, 8, 56, 696, 373, 380, 459, 478,
            162, 329, 218, 570, 483, 485, 441, 601, 498, 356, 580, 151, 445, 599, 99, 503,
            396, 741, 764, 234, 110, 540, 738, 13, 40, 176, 438, 550, 65, 355, 287, 57,
            289, 134, 179, 431, 17, 713, 486, 408, 707, 182, 207, 525, 278, 638, 475, 67,
            681, 271, 404, 692, 476, 193, 293, 501, 465, 4, 616, 715, 313, 506, 70, 415,
            427, 94, 230, 402, 504, 492, 565, 761, 510, 243, 555, 344, 35, 572, 597, 314,
            136, 145, 23, 723, 644, 545, 282, 642, 180, 528, 676, 63, 102, 126, 750, 118,
            719, 509, 292, 389, 544, 573, 430, 74, 270, 160, 239, 170, 516, 122, 517, 247,
            419, 624, 654, 758, 663, 92, 374, 319, 131, 120, 252, 674, 449, 264, 639, 484,
            6, 412, 553, 488, 541, 321, 467, 185, 711, 281, 700, 429, 290, 493, 426, 146,
            45, 383, 683, 64, 288, 52, 640, 295, 641, 273, 684, 620, 112, 244, 150, 166,
            238, 33, 697, 119, 575, 317, 390, 310, 466, 479, 519, 743, 671, 705, 107, 495,
            306, 251, 385, 257, 24, 48, 54, 167, 284, 604, 538, 301, 60, 456, 66, 197,
            138, 387, 175, 546, 610, 443, 214, 153, 688, 463, 358, 569, 258, 398, 444, 139,
            543, 116, 535, 633, 365, 729, 734, 450, 342, 269, 183, 452, 656, 422, 439, 311,
            619, 691, 628, 709, 38, 636, 108, 648, 5, 710, 420, 635, 241, 581, 11, 103,
            524, 216, 582, 435, 388, 437, 557, 515, 73, 552, 86, 462, 20, 318, 111, 421,
            453, 751, 395, 594, 451, 747, 614, 425, 203, 455, 128, 520, 53, 621, 12, 259,
            673, 315, 763, 18, 366, 156, 693, 154, 703, 308, 634, 75, 563, 382, 680, 587,
            169, 556, 222, 379, 698, 744, 27, 215, 10, 327, 49, 767, 300, 186, 392, 527,
            113, 473, 190, 482, 708, 337, 275, 428, 447, 414, 276, 647, 213, 250, 489, 133,
            602, 68, 615, 245, 755, 371, 95, 551, 199, 263, 694, 339, 722, 487, 260, 79,
            285, 304, 672, 212, 653, 568, 50, 192, 265, 409, 274, 105, 233, 378, 211, 61,
            411, 712, 588, 181, 123, 521, 340, 246, 606, 658, 352, 235, 205, 735, 277, 730,
            643, 760, 78, 595, 242, 477, 72, 36, 353, 76, 542, 608, 536, 163, 343, 141,
            496, 384, 84, 491, 28, 576, 115, 645, 369, 130, 666, 237, 593, 303, 701, 617,
            579, 454, 731, 208, 25, 561, 178, 757, 461, 695, 590, 381, 762, 106, 299, 652,
            46, 566, 361, 148, 232, 737, 93, 745, 458, 312, 752, 500, 625, 83, 584, 649,
            189, 363, 637, 164, 22, 117, 32, 37, 307, 325, 268, 626, 718, 357, 350, 564,
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
