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
            116, 851, 442, 848, 907, 913, 928, 396, 926, 954, 360, 579, 173, 671, 537, 518,
            922, 21, 319, 32, 561, 650, 206, 73, 475, 224, 328, 751, 596, 269, 301, 732,
            644, 185, 55, 437, 427, 880, 480, 13, 277, 438, 212, 373, 863, 821, 934, 64,
            266, 308, 526, 386, 104, 33, 766, 202, 170, 570, 942, 154, 422, 957, 558, 802,
            196, 834, 305, 416, 728, 629, 630, 994, 176, 329, 953, 218, 521, 389, 757, 228,
            976, 980, 804, 920, 566, 213, 204, 626, 321, 685, 298, 303, 649, 93, 564, 797,
            130, 860, 617, 479, 187, 77, 310, 753, 915, 704, 273, 785, 424, 819, 383, 125,
            999, 667, 806, 238, 5, 1001, 883, 619, 419, 955, 320, 565, 115, 634, 861, 576,
            681, 714, 614, 824, 487, 737, 622, 642, 659, 555, 749, 748, 601, 1021, 914, 22,
            251, 76, 676, 194, 457, 167, 453, 462, 287, 175, 893, 259, 1002, 464, 493, 866,
            1014, 11, 296, 933, 379, 532, 450, 409, 265, 486, 2, 825, 859, 369, 88, 964,
            520, 512, 502, 688, 995, 972, 394, 683, 874, 726, 582, 148, 546, 143, 248, 114,
            1011, 946, 349, 800, 910, 113, 485, 140, 655, 24, 4, 662, 772, 51, 1022, 347,
            215, 940, 375, 387, 231, 129, 799, 94, 831, 727, 575, 508, 663, 135, 20, 59,
            1009, 723, 624, 713, 786, 100, 736, 352, 82, 52, 877, 404, 152, 774, 392, 49,
            911, 654, 592, 639, 898, 517, 965, 703, 252, 455, 924, 344, 99, 188, 643, 690,
            849, 616, 687, 725, 439, 286, 69, 653, 646, 307, 230, 262, 612, 232, 142, 306,
            469, 279, 891, 283, 92, 84, 729, 235, 513, 356, 200, 742, 969, 645, 857, 324,
            183, 1017, 159, 793, 117, 370, 586, 917, 966, 239, 72, 752, 163, 102, 312, 854,
            588, 86, 397, 309, 500, 528, 253, 638, 943, 353, 722, 741, 641, 927, 686, 826,
            760, 549, 571, 112, 838, 783, 338, 992, 458, 871, 470, 552, 192, 168, 107, 501,
            426, 670, 677, 166, 573, 665, 803, 36, 236, 765, 531, 853, 810, 28, 569, 856,
            511, 504, 146, 441, 388, 544, 268, 507, 608, 61, 868, 719, 572, 648, 459, 180,
            66, 208, 258, 401, 57, 603, 795, 993, 692, 162, 739, 366, 9, 8, 897, 300,
            669, 699, 590, 724, 134, 6, 832, 219, 381, 374, 820, 473, 186, 776, 661, 304,
            944, 691, 717, 420, 385, 833, 609, 267, 71, 264, 403, 492, 85, 43, 358, 948,
            79, 1020, 873, 967, 288, 367, 775, 178, 19, 315, 697, 782, 293, 294, 986, 708,
            949, 408, 540, 840, 892, 975, 829, 996, 337, 510, 706, 193, 574, 361, 149, 98,
            153, 400, 335, 42, 758, 23, 466, 471, 103, 119, 983, 931, 543, 862, 207, 285,
            527, 998, 406, 689, 698, 530, 425, 884, 591, 14, 91, 497, 160, 896, 909, 62,
            767, 169, 341, 282, 580, 281, 631, 875, 198, 371, 822, 181, 357, 968, 529, 60,
            243, 378, 461, 524, 495, 247, 390, 837, 912, 195, 885, 263, 302, 982, 78, 226,
            363, 780, 519, 563, 585, 850, 534, 40, 205, 899, 827, 110, 272, 330, 754, 977,
            354, 157, 864, 280, 509, 221, 664, 1005, 551, 240, 858, 1000, 210, 465, 405, 715,
            984, 488, 244, 970, 256, 621, 545, 70, 554, 506, 377, 548, 63, 382, 46, 759,
            1004, 189, 636, 547, 452, 541, 906, 542, 313, 788, 241, 111, 1012, 227, 190, 234,
            463, 295, 498, 743, 54, 89, 962, 435, 316, 789, 384, 50, 835, 745, 372, 587,
            895, 83, 490, 981, 494, 613, 399, 209, 292, 474, 932, 410, 423, 150, 979, 211,
            625, 138, 632, 432, 299, 901, 137, 318, 289, 808, 656, 161, 136, 1007, 929, 764,
            887, 652, 398, 550, 334, 496, 985, 151, 919, 327, 1003, 607, 734, 925, 660, 291,
            440, 935, 770, 989, 923, 747, 229, 523, 172, 106, 593, 380, 876, 904, 578, 675,
            807, 123, 900, 1016, 31, 990, 203, 436, 133, 53, 771, 744, 365, 413, 567, 709,
            322, 869, 359, 350, 417, 610, 7, 635, 489, 841, 451, 97, 842, 801, 431, 124,
            355, 952, 456, 75, 668, 182, 222, 959, 628, 216, 700, 1006, 448, 407, 249, 362,
            716, 750, 997, 937, 879, 882, 428, 391, 730, 121, 101, 145, 811, 693, 602, 284,
            444, 445, 903, 12, 255, 637, 605, 261, 26, 881, 39, 679, 393, 1018, 615, 339,
            275, 0, 973, 323, 242, 855, 278, 74, 779, 430, 478, 171, 1019, 412, 331, 233,
            271, 598, 274, 447, 890, 738, 1013, 415, 17, 974, 813, 701, 481, 333, 905, 68,
            559, 845, 1, 599, 56, 346, 538, 756, 414, 878, 351, 454, 27, 791, 1008, 740,
            467, 225, 120, 254, 38, 95, 332, 499, 191, 746, 963, 568, 477, 945, 90, 651,
            694, 809, 894, 978, 584, 418, 108, 476, 938, 214, 682, 888, 794, 139, 67, 1015,
            680, 184, 131, 317, 411, 1023, 839, 886, 577, 658, 583, 918, 44, 971, 828, 702,
            126, 600, 777, 902, 847, 177, 640, 987, 515, 805, 560, 342, 595, 763, 30, 947,
            961, 525, 245, 311, 217, 250, 594, 816, 684, 562, 720, 118, 158, 711, 144, 916,
            781, 45, 201, 865, 260, 482, 472, 830, 815, 769, 823, 460, 10, 132, 939, 257,
            844, 237, 348, 852, 950, 710, 768, 773, 505, 707, 836, 533, 535, 633, 790, 174,
            611, 672, 343, 58, 246, 792, 991, 812, 673, 87, 483, 48, 491, 941, 956, 627,
            731, 581, 889, 433, 666, 25, 276, 421, 336, 516, 936, 127, 870, 761, 29, 695,
            762, 109, 620, 522, 443, 735, 34, 155, 604, 484, 164, 105, 778, 657, 755, 156,
            325, 65, 539, 368, 623, 678, 376, 297, 556, 345, 35, 814, 958, 3, 705, 402,
            908, 147, 270, 721, 843, 15, 557, 468, 597, 449, 817, 290, 733, 553, 16, 326,
            536, 446, 199, 796, 220, 128, 872, 395, 787, 47, 41, 434, 429, 784, 37, 1010,
            930, 81, 618, 514, 18, 589, 503, 846, 606, 712, 165, 921, 340, 818, 179, 141,
            674, 696, 960, 122, 798, 223, 867, 988, 718, 197, 364, 96, 647, 314, 80, 951,
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
