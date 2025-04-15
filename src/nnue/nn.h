#pragma once

#define COUNT_FOR_PERMUTATIONS 1
#undef COUNT_FOR_PERMUTATIONS

#define NO_WEIGHT_PERMUTING 1
#undef NO_WEIGHT_PERMUTING

#include "../defs.h"
#include "../nnue/arch.h"
#include "../position.h"
#include "accumulator.h"

#include <array>
#include <span>

namespace Horsie::NNUE {

#if defined(COUNT_FOR_PERMUTATIONS)
    extern std::array<u64, L1_SIZE> NNZCounts;
    extern u64 ActivationCount;
    extern u64 EvalCalls;
#endif

    template<typename T>
    using Span = std::span<T>;

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
        std::array<T, INPUT_SIZE * L1_SIZE * INPUT_BUCKETS> FTWeights;
        std::array<T, L1_SIZE>                              FTBiases;
        Util::NDArray<W, OUTPUT_BUCKETS, L1_SIZE * L2_SIZE> L1Weights;
        Util::NDArray<U, OUTPUT_BUCKETS, L2_SIZE>           L1Biases;
        Util::NDArray<U, OUTPUT_BUCKETS, L2_SIZE * L3_SIZE> L2Weights;
        Util::NDArray<U, OUTPUT_BUCKETS, L3_SIZE>           L2Biases;
        Util::NDArray<U, OUTPUT_BUCKETS, L3_SIZE>           L3Weights;
        std::array<U, OUTPUT_BUCKETS>                       L3Biases;
    };
    using Network = NetworkBase<i16, i8, float>;


    struct alignas(64) NNZTable {
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
        328, 992, 699, 542, 11, 370, 240, 874, 674, 534, 631, 423, 446, 355, 982, 376,
        833, 325, 284, 528, 147, 927, 324, 473, 577, 686, 808, 39, 0, 215, 931, 359,
        180, 862, 1005, 430, 253, 835, 775, 206, 920, 226, 52, 888, 995, 730, 118, 278,
        605, 584, 811, 772, 429, 225, 112, 624, 105, 168, 6, 967, 456, 925, 844, 810,
        273, 146, 185, 890, 971, 19, 828, 778, 268, 35, 878, 279, 186, 317, 673, 993,
        947, 214, 801, 195, 374, 949, 589, 261, 532, 132, 755, 991, 733, 543, 919, 939,
        83, 806, 130, 509, 265, 127, 610, 177, 510, 764, 262, 99, 582, 108, 1009, 455,
        353, 169, 379, 892, 911, 574, 526, 220, 272, 218, 485, 210, 152, 831, 417, 203,
        652, 102, 454, 403, 648, 816, 585, 66, 56, 524, 819, 228, 973, 829, 305, 752,
        905, 912, 14, 786, 917, 689, 784, 609, 687, 985, 77, 719, 740, 672, 107, 704,
        165, 384, 978, 646, 373, 474, 445, 754, 669, 338, 435, 159, 25, 378, 42, 135,
        452, 545, 254, 308, 891, 80, 737, 149, 256, 614, 956, 697, 600, 24, 872, 791,
        201, 224, 257, 5, 298, 746, 442, 567, 945, 33, 930, 314, 710, 558, 515, 75,
        287, 432, 580, 231, 459, 461, 1012, 647, 952, 61, 521, 255, 814, 207, 89, 163,
        840, 352, 161, 607, 701, 234, 638, 47, 997, 512, 98, 449, 110, 866, 777, 915,
        309, 470, 536, 527, 873, 970, 922, 419, 491, 74, 139, 7, 896, 299, 535, 660,
        158, 48, 518, 404, 205, 924, 497, 348, 684, 869, 843, 250, 602, 926, 150, 906,
        85, 176, 22, 824, 629, 54, 757, 329, 637, 650, 581, 44, 865, 651, 443, 690,
        472, 368, 166, 119, 1016, 708, 720, 311, 45, 242, 658, 347, 898, 802, 221, 634,
        628, 880, 836, 692, 114, 688, 479, 438, 187, 747, 571, 592, 618, 712, 965, 679,
        959, 789, 709, 793, 460, 666, 694, 748, 691, 188, 941, 160, 18, 942, 575, 657,
        138, 463, 151, 492, 665, 773, 988, 189, 519, 293, 887, 251, 854, 830, 1004, 670,
        409, 484, 486, 148, 974, 402, 318, 736, 951, 396, 758, 817, 1022, 871, 794, 766,
        820, 356, 676, 239, 972, 310, 371, 860, 547, 929, 59, 216, 914, 695, 1011, 157,
        481, 405, 200, 36, 675, 626, 564, 346, 336, 382, 803, 129, 49, 1018, 953, 457,
        725, 483, 565, 608, 964, 68, 508, 902, 913, 16, 319, 560, 799, 86, 363, 578,
        1000, 53, 281, 140, 2, 191, 877, 55, 718, 776, 598, 563, 743, 343, 67, 1002,
        848, 390, 389, 156, 897, 196, 245, 882, 344, 62, 728, 594, 369, 476, 333, 199,
        742, 385, 244, 530, 171, 504, 183, 500, 966, 761, 395, 583, 462, 731, 546, 217,
        886, 145, 104, 1013, 961, 950, 804, 362, 173, 539, 291, 863, 285, 100, 467, 307,
        90, 502, 642, 750, 627, 895, 826, 3, 870, 393, 523, 715, 529, 76, 306, 315,
        345, 981, 91, 63, 968, 243, 154, 367, 495, 842, 759, 439, 599, 128, 190, 693,
        640, 178, 120, 436, 727, 313, 398, 121, 26, 361, 111, 164, 533, 50, 399, 323,
        823, 31, 606, 767, 274, 753, 935, 92, 230, 716, 70, 141, 375, 377, 857, 143,
        852, 15, 32, 426, 269, 482, 938, 636, 859, 12, 800, 340, 487, 603, 407, 365,
        237, 907, 321, 683, 732, 113, 790, 122, 133, 155, 858, 805, 782, 408, 601, 998,
        682, 723, 334, 499, 934, 933, 619, 549, 440, 918, 267, 632, 552, 326, 876, 364,
        986, 223, 43, 322, 655, 289, 1014, 366, 937, 471, 901, 923, 780, 1015, 88, 957,
        331, 20, 825, 621, 739, 900, 576, 425, 698, 904, 248, 351, 932, 65, 883, 103,
        235, 783, 292, 513, 969, 762, 962, 994, 812, 987, 172, 990, 954, 707, 958, 729,
        975, 756, 480, 548, 283, 785, 123, 397, 760, 411, 1017, 795, 641, 490, 498, 302,
        586, 51, 792, 212, 388, 983, 410, 908, 232, 41, 885, 131, 613, 229, 622, 522,
        943, 726, 332, 566, 781, 46, 401, 21, 517, 573, 774, 864, 1010, 441, 101, 839,
        662, 615, 37, 282, 722, 162, 597, 847, 721, 339, 853, 277, 717, 300, 153, 705,
        8, 645, 568, 656, 680, 813, 744, 955, 87, 1023, 219, 372, 765, 881, 204, 465,
        856, 415, 448, 428, 579, 556, 275, 763, 124, 738, 387, 706, 751, 649, 550, 846,
        427, 179, 596, 246, 849, 489, 38, 64, 834, 117, 815, 354, 271, 94, 276, 587,
        134, 541, 84, 418, 28, 96, 93, 209, 496, 412, 301, 788, 258, 175, 142, 633,
        861, 252, 40, 948, 940, 17, 903, 702, 851, 280, 488, 34, 977, 202, 960, 9,
        327, 1006, 537, 464, 677, 538, 989, 95, 1008, 659, 735, 771, 714, 106, 72, 71,
        192, 1019, 320, 671, 711, 544, 115, 360, 57, 380, 381, 616, 69, 416, 125, 342,
        696, 593, 1021, 211, 27, 668, 561, 208, 312, 227, 182, 630, 13, 916, 928, 724,
        58, 741, 413, 946, 29, 768, 617, 936, 787, 867, 335, 60, 531, 383, 516, 681,
        144, 137, 976, 304, 263, 893, 899, 170, 431, 126, 421, 827, 350, 590, 837, 288,
        505, 198, 635, 822, 562, 444, 845, 23, 569, 167, 559, 330, 879, 884, 664, 294,
        341, 477, 466, 506, 434, 540, 1001, 551, 809, 290, 625, 259, 236, 469, 894, 745,
        73, 832, 1, 386, 357, 136, 81, 394, 478, 555, 553, 468, 349, 514, 818, 494,
        980, 270, 661, 643, 437, 889, 807, 420, 4, 392, 525, 433, 241, 475, 447, 797,
        222, 572, 295, 875, 358, 1007, 197, 612, 623, 451, 821, 193, 174, 796, 639, 30,
        963, 703, 838, 868, 570, 116, 249, 97, 770, 850, 194, 595, 999, 453, 247, 588,
        10, 264, 1020, 391, 749, 734, 944, 520, 910, 414, 458, 979, 604, 557, 501, 503,
        184, 422, 611, 316, 654, 620, 644, 667, 82, 266, 653, 181, 798, 286, 769, 984,
        678, 909, 260, 591, 713, 663, 554, 511, 297, 303, 685, 79, 779, 109, 507, 233,
        213, 424, 400, 855, 450, 841, 996, 78, 337, 296, 700, 1003, 921, 238, 493, 406,
    };

    constexpr auto PermuteIndices = [] {
        std::array<i32, L1_PAIR_COUNT> arr{};

        for (i32 i = 0; i < L1_PAIR_COUNT; ++i) {
#if defined(NO_WEIGHT_PERMUTING)
            arr[i] = i;
#else
            arr[i] = BestPermuteIndices[i];
#endif
        }

        return arr;
    }();

}
