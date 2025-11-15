#pragma once

#define PERM_COUNT 1
#undef PERM_COUNT

#define NO_WEIGHT_PERMUTING 1
#undef NO_WEIGHT_PERMUTING

#include "../defs.h"
#include "../nnue/arch.h"
#include "../position.h"
#include "accumulator.h"

#include <array>
#include <span>

namespace Horsie::NNUE {

#if defined(PERM_COUNT)
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
        std::array<__m128i, 256> Entries;
    };

    extern Network net;
    extern NNZTable nnzTable;

    bool IsCompressed(std::istream& stream);
    void LoadZSTD(std::istream& m_stream, std::byte* dst);
    void LoadNetwork(const std::string& name);
    void SetupNNZ();
    void PermuteFT(Span<i16> ftWeights, Span<i16> ftBiases);
    void PermuteL1(i8 l1Weights[L1_SIZE][OUTPUT_BUCKETS][L2_SIZE]);

    i32 GetEvaluation(Position& pos, i32 outputBucket);
    i32 GetEvaluation(Position& pos);

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

    void Sub(const i16* src, i16* dst, const i16* sub1);
    void Add(const i16* src, i16* dst, const i16* add1);
    void SubAdd(const i16* src, i16* dst, const i16* sub1, const i16* add1);
    void SubSubAdd(const i16* src, i16* dst, const i16* sub1, const i16* sub2, const i16* add1);
    void SubSubAddAdd(const i16* src, i16* dst, const i16* sub1, const i16* sub2, const i16* add1, const i16* add2);


    constexpr i32 BestPermuteIndices[] = {
        238, 406, 493, 921, 296, 700, 1003, 855, 663, 450, 996, 511, 337, 400, 82, 424,
        841, 984, 769, 554, 653, 181, 678, 286, 266, 667, 109, 604, 78, 303, 713, 798,
        213, 889, 644, 79, 685, 507, 850, 838, 999, 909, 184, 503, 591, 557, 770, 501,
        620, 260, 654, 779, 97, 595, 420, 724, 514, 944, 263, 868, 297, 500, 316, 359,
        797, 879, 386, 9, 699, 453, 570, 73, 568, 69, 596, 194, 198, 330, 916, 807,
        126, 910, 1007, 475, 494, 734, 427, 414, 936, 822, 542, 81, 349, 197, 71, 818,
        360, 170, 4, 437, 703, 193, 645, 125, 447, 312, 95, 588, 10, 264, 27, 505,
        433, 982, 116, 233, 72, 992, 504, 270, 241, 827, 436, 458, 358, 550, 561, 832,
        275, 295, 1023, 342, 611, 183, 928, 851, 751, 899, 247, 549, 446, 753, 771, 167,
        612, 963, 136, 849, 635, 328, 134, 617, 768, 621, 431, 735, 294, 422, 639, 958,
        355, 980, 630, 1020, 661, 392, 357, 174, 391, 931, 664, 372, 469, 1, 171, 623,
        428, 633, 553, 787, 237, 60, 572, 569, 796, 208, 593, 122, 540, 745, 92, 94,
        421, 236, 551, 8, 301, 246, 643, 29, 884, 668, 544, 726, 576, 837, 728, 464,
        288, 259, 249, 811, 867, 350, 13, 468, 616, 763, 520, 628, 302, 196, 16, 0,
        175, 325, 466, 809, 215, 859, 142, 656, 58, 209, 960, 473, 410, 179, 30, 893,
        488, 815, 340, 416, 875, 525, 904, 387, 823, 1006, 96, 23, 439, 31, 606, 895,
        38, 702, 245, 813, 537, 597, 388, 979, 229, 881, 347, 188, 590, 587, 836, 380,
        506, 204, 335, 327, 619, 1001, 252, 605, 988, 897, 412, 489, 106, 418, 324, 154,
        17, 498, 32, 714, 833, 528, 636, 367, 15, 51, 14, 45, 160, 332, 935, 559,
        20, 711, 648, 968, 477, 541, 321, 556, 518, 448, 57, 555, 434, 202, 19, 934,
        749, 120, 788, 222, 579, 480, 933, 84, 223, 258, 946, 274, 649, 182, 62, 290,
        729, 894, 153, 101, 989, 853, 864, 137, 976, 954, 940, 821, 124, 415, 522, 66,
        780, 394, 858, 331, 322, 707, 826, 180, 508, 805, 825, 115, 696, 339, 584, 977,
        562, 792, 647, 677, 1008, 756, 578, 12, 269, 941, 983, 778, 516, 143, 43, 907,
        425, 846, 471, 665, 117, 63, 478, 28, 722, 900, 786, 277, 882, 144, 842, 284,
        21, 594, 905, 190, 232, 772, 262, 465, 659, 351, 938, 444, 834, 795, 219, 496,
        227, 794, 48, 426, 162, 548, 242, 145, 451, 155, 354, 955, 251, 698, 381, 538,
        64, 86, 248, 640, 470, 845, 804, 691, 374, 404, 876, 280, 912, 147, 641, 903,
        211, 1000, 913, 739, 34, 283, 1011, 123, 957, 1010, 341, 87, 375, 371, 948, 377,
        1015, 441, 326, 552, 627, 607, 586, 800, 923, 345, 759, 111, 732, 276, 363, 467,
        692, 1021, 362, 364, 300, 24, 91, 440, 212, 560, 885, 113, 129, 871, 886, 88,
        35, 228, 766, 927, 1009, 527, 799, 687, 378, 793, 975, 987, 271, 950, 765, 138,
        775, 463, 592, 706, 1013, 483, 178, 748, 539, 566, 133, 361, 998, 99, 738, 376,
        721, 599, 961, 883, 243, 932, 26, 41, 1014, 291, 495, 93, 310, 365, 760, 482,
        783, 589, 42, 172, 1017, 828, 773, 166, 803, 959, 148, 812, 230, 672, 622, 1005,
        930, 781, 334, 531, 383, 750, 671, 411, 573, 727, 571, 613, 774, 176, 785, 547,
        937, 390, 725, 158, 880, 887, 187, 972, 385, 314, 59, 533, 157, 860, 65, 261,
        642, 318, 695, 764, 1004, 658, 974, 389, 717, 396, 85, 472, 486, 461, 449, 265,
        121, 861, 319, 313, 502, 943, 407, 716, 994, 490, 839, 462, 455, 565, 743, 151,
        929, 862, 484, 986, 107, 683, 281, 681, 37, 693, 53, 676, 323, 789, 874, 46,
        603, 863, 510, 949, 366, 289, 235, 491, 408, 819, 852, 962, 405, 546, 816, 723,
        758, 848, 535, 720, 790, 108, 40, 401, 817, 524, 513, 791, 285, 981, 189, 200,
        487, 682, 1022, 499, 307, 744, 103, 762, 625, 517, 847, 119, 156, 1018, 767, 255,
        304, 104, 3, 459, 877, 292, 161, 709, 519, 890, 997, 1019, 128, 873, 239, 680,
        747, 413, 481, 305, 399, 856, 914, 951, 397, 149, 492, 49, 869, 964, 336, 990,
        741, 608, 100, 150, 824, 1012, 670, 368, 947, 140, 240, 718, 409, 127, 840, 430,
        343, 908, 139, 89, 173, 317, 135, 598, 67, 70, 534, 130, 11, 708, 114, 293,
        704, 56, 705, 74, 892, 452, 536, 402, 694, 131, 808, 956, 370, 68, 257, 191,
        479, 684, 610, 829, 432, 601, 993, 250, 679, 309, 192, 512, 754, 971, 857, 742,
        564, 55, 878, 1016, 906, 5, 7, 782, 98, 650, 267, 712, 740, 50, 1002, 141,
        655, 186, 164, 673, 736, 2, 382, 801, 344, 697, 872, 820, 206, 563, 195, 830,
        776, 395, 843, 308, 44, 710, 615, 419, 320, 945, 457, 583, 854, 61, 90, 626,
        199, 529, 523, 132, 201, 614, 76, 268, 47, 634, 582, 279, 970, 152, 254, 731,
        675, 969, 217, 618, 398, 22, 311, 515, 526, 926, 602, 225, 185, 666, 844, 746,
        315, 637, 902, 646, 600, 918, 965, 953, 733, 384, 662, 393, 761, 352, 686, 474,
        752, 530, 165, 220, 558, 224, 715, 985, 922, 253, 629, 379, 810, 737, 203, 346,
        973, 609, 105, 925, 306, 216, 445, 898, 911, 244, 967, 891, 177, 163, 356, 580,
        901, 831, 102, 581, 6, 896, 33, 835, 460, 39, 25, 435, 36, 77, 52, 205,
        110, 521, 757, 802, 168, 915, 688, 118, 210, 755, 545, 329, 282, 214, 784, 333,
        632, 660, 80, 256, 777, 652, 942, 443, 567, 83, 624, 54, 75, 651, 509, 730,
        669, 417, 169, 952, 438, 442, 476, 991, 369, 456, 429, 146, 719, 348, 966, 273,
        978, 353, 221, 888, 575, 454, 159, 631, 870, 689, 920, 234, 287, 543, 532, 814,
        577, 574, 338, 806, 299, 231, 585, 112, 373, 701, 218, 272, 919, 638, 485, 657,
        207, 690, 403, 917, 866, 939, 278, 423, 674, 497, 298, 226, 18, 865, 995, 924,
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
