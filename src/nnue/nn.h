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
            349, 553, 739, 409, 491, 735, 711, 579, 694, 188, 339, 444, 254, 454, 680, 353,
            54, 520, 75, 205, 155, 308, 275, 581, 551, 706, 511, 222, 733, 749, 400, 195,
            556, 58, 449, 45, 715, 583, 488, 47, 46, 361, 500, 398, 474, 598, 764, 709,
            588, 670, 508, 689, 528, 728, 141, 563, 561, 460, 360, 470, 218, 326, 717, 590,
            83, 645, 225, 669, 204, 185, 189, 350, 660, 659, 28, 194, 315, 390, 317, 285,
            647, 546, 515, 686, 43, 462, 623, 68, 364, 171, 88, 365, 536, 423, 21, 287,
            260, 55, 412, 228, 542, 626, 677, 376, 371, 429, 146, 716, 130, 673, 281, 476,
            316, 1, 366, 538, 422, 233, 646, 29, 217, 267, 654, 748, 115, 635, 606, 648,
            103, 208, 35, 114, 393, 296, 517, 566, 3, 549, 348, 640, 344, 701, 564, 541,
            745, 437, 64, 362, 25, 52, 436, 249, 425, 158, 650, 300, 502, 49, 107, 730,
            126, 14, 258, 79, 176, 252, 65, 574, 624, 490, 693, 149, 712, 649, 97, 214,
            69, 420, 475, 268, 593, 18, 487, 414, 85, 74, 255, 184, 451, 82, 95, 354,
            122, 691, 492, 341, 298, 342, 620, 405, 600, 234, 84, 402, 721, 118, 643, 592,
            322, 573, 627, 708, 597, 172, 321, 524, 567, 244, 668, 331, 358, 383, 533, 333,
            736, 724, 57, 416, 570, 482, 53, 591, 760, 319, 311, 379, 505, 671, 17, 392,
            382, 752, 483, 178, 263, 162, 261, 297, 700, 277, 510, 705, 309, 559, 150, 682,
            137, 713, 279, 211, 743, 318, 241, 608, 355, 747, 352, 100, 119, 740, 219, 179,
            415, 571, 271, 663, 11, 102, 644, 537, 450, 23, 293, 661, 312, 548, 113, 190,
            180, 283, 288, 324, 41, 755, 78, 154, 690, 432, 135, 381, 98, 658, 678, 417,
            753, 403, 506, 656, 209, 253, 104, 585, 215, 461, 484, 340, 495, 397, 270, 389,
            418, 42, 639, 120, 345, 27, 372, 201, 589, 306, 66, 746, 445, 187, 404, 248,
            173, 519, 166, 385, 455, 282, 630, 36, 438, 531, 284, 679, 493, 607, 435, 347,
            134, 330, 232, 13, 110, 227, 578, 186, 375, 246, 496, 73, 15, 159, 687, 197,
            507, 212, 148, 486, 410, 594, 653, 672, 394, 336, 530, 534, 652, 732, 615, 637,
            391, 744, 723, 695, 729, 458, 224, 722, 684, 343, 369, 183, 111, 501, 96, 265,
            329, 62, 512, 464, 94, 472, 257, 132, 452, 407, 762, 164, 580, 199, 555, 676,
            191, 544, 192, 203, 430, 642, 618, 138, 577, 616, 240, 565, 522, 123, 754, 332,
            440, 337, 550, 213, 386, 489, 81, 378, 86, 667, 136, 509, 105, 572, 477, 198,
            664, 413, 568, 370, 725, 595, 683, 144, 153, 539, 206, 256, 121, 247, 765, 584,
            582, 545, 10, 87, 139, 662, 295, 535, 688, 8, 338, 759, 480, 127, 726, 610,
            165, 294, 625, 323, 220, 527, 521, 250, 210, 31, 636, 147, 269, 39, 613, 448,
            727, 605, 286, 207, 633, 221, 586, 92, 304, 30, 357, 634, 742, 356, 145, 5,
            665, 478, 116, 310, 473, 707, 638, 251, 131, 681, 766, 174, 441, 602, 142, 396,
            720, 129, 631, 526, 467, 196, 611, 466, 313, 334, 601, 758, 532, 200, 302, 657,
            61, 63, 280, 710, 91, 651, 89, 763, 19, 133, 387, 229, 231, 421, 446, 406,
            181, 278, 666, 290, 599, 328, 388, 59, 236, 702, 419, 289, 56, 377, 447, 169,
            408, 140, 374, 259, 465, 37, 305, 243, 7, 112, 168, 587, 76, 632, 427, 303,
            156, 299, 226, 22, 655, 32, 756, 128, 703, 612, 170, 469, 738, 459, 239, 223,
            714, 373, 481, 48, 0, 367, 750, 245, 698, 9, 272, 494, 443, 12, 457, 453,
            262, 4, 734, 737, 380, 562, 767, 463, 77, 434, 242, 291, 51, 99, 363, 216,
            699, 540, 20, 6, 2, 60, 516, 497, 109, 177, 697, 182, 264, 125, 576, 485,
            274, 558, 24, 504, 596, 143, 498, 603, 90, 431, 273, 93, 503, 307, 72, 525,
            33, 569, 479, 124, 621, 547, 518, 557, 395, 575, 71, 351, 230, 411, 439, 741,
            276, 368, 16, 151, 560, 40, 614, 523, 622, 359, 529, 428, 674, 292, 167, 314,
            692, 327, 202, 235, 401, 719, 704, 641, 161, 238, 609, 513, 685, 67, 617, 163,
            26, 468, 456, 335, 70, 751, 471, 514, 399, 554, 50, 696, 628, 757, 552, 499,
            157, 629, 44, 160, 424, 117, 675, 80, 346, 619, 108, 34, 237, 731, 193, 442,
            384, 152, 761, 426, 543, 106, 266, 320, 718, 301, 38, 604, 433, 175, 325, 101,
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
