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
    struct alignas(64) RawNetworkBase {
        T FTWeights[INPUT_SIZE * L1_SIZE * INPUT_BUCKETS];
        T FTBiases[L1_SIZE];
        W L1Weights[OUTPUT_BUCKETS][L2_SIZE][L1_SIZE];
        U L1Biases[OUTPUT_BUCKETS][L2_SIZE];
        U L2Weights[OUTPUT_BUCKETS][L3_SIZE][L2_SIZE];
        U L2Biases[OUTPUT_BUCKETS][L3_SIZE];
        U L3Weights[OUTPUT_BUCKETS][L3_SIZE];
        U L3Biases[OUTPUT_BUCKETS];
    };
    using RawNetwork = RawNetworkBase<i16, i8, float>;

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
    static void PermuteL1(i8 l1Weights[OUTPUT_BUCKETS][L2_SIZE][L1_SIZE]);


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


    constexpr i32 BucketScheme[] = {
        0, 1, 2, 3,
        4, 5, 6, 7,
        4, 5, 6, 7,
        8, 8, 8, 8,
        8, 8, 8, 8,
        9, 9, 9, 9,
        9, 9, 9, 9,
        9, 9, 9, 9,
    };

    constexpr auto KingBuckets = []
    {
        std::array<i32, 64> buckets{};

        const auto nBuckets = *std::max_element(BucketScheme, &BucketScheme[32]) + 1;
        for (i32 rank = 0; rank < 8; rank++) {
            for (i32 file = 0; file < 4; file++) {
                const auto b = BucketScheme[rank * 4 + file];
                const auto dst = rank * 8 + file;

                buckets[dst] = b;
                buckets[dst ^ 7] = b + nBuckets;
            }
        }

        return buckets;
    }();


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
        482, 305, 155, 233, 10, 39, 663, 710, 83, 720, 26, 217, 559, 515, 411, 75,
        380, 42, 418, 462, 451, 140, 333, 660, 586, 563, 185, 174, 103, 356, 670, 293,
        107, 129, 596, 538, 105, 640, 407, 220, 227, 562, 278, 16, 244, 414, 608, 694,
        592, 662, 297, 502, 136, 195, 649, 712, 149, 400, 477, 250, 422, 399, 496, 746,
        231, 376, 183, 363, 335, 135, 555, 450, 497, 639, 724, 690, 73, 85, 8, 518,
        752, 683, 719, 18, 751, 625, 222, 273, 245, 15, 429, 374, 213, 529, 270, 667,
        403, 256, 439, 142, 572, 504, 151, 187, 264, 208, 488, 63, 6, 13, 172, 644,
        698, 296, 271, 632, 430, 340, 218, 552, 306, 759, 255, 9, 545, 371, 378, 626,
        162, 636, 706, 37, 167, 666, 711, 551, 621, 738, 716, 47, 175, 57, 275, 610,
        389, 673, 615, 236, 491, 357, 472, 622, 597, 163, 527, 198, 166, 272, 650, 178,
        638, 753, 729, 426, 216, 160, 386, 609, 541, 209, 158, 238, 362, 118, 576, 181,
        664, 495, 756, 703, 223, 132, 456, 35, 341, 45, 322, 171, 655, 398, 574, 532,
        611, 321, 312, 448, 76, 469, 500, 416, 489, 123, 53, 188, 40, 671, 440, 553,
        145, 695, 168, 544, 352, 547, 329, 46, 96, 528, 394, 533, 630, 568, 526, 202,
        599, 623, 44, 606, 402, 494, 280, 406, 682, 445, 709, 457, 455, 397, 566, 350,
        284, 316, 618, 732, 687, 206, 7, 359, 141, 635, 290, 182, 461, 575, 387, 633,
        643, 89, 70, 766, 723, 641, 607, 106, 471, 299, 485, 404, 454, 603, 419, 424,
        601, 696, 366, 624, 144, 475, 32, 133, 55, 353, 530, 190, 84, 253, 479, 681,
        464, 153, 531, 266, 605, 379, 77, 739, 512, 54, 628, 152, 332, 459, 413, 130,
        25, 686, 415, 382, 29, 583, 743, 590, 67, 331, 94, 225, 591, 114, 651, 505,
        390, 543, 323, 513, 747, 154, 678, 285, 665, 578, 396, 186, 594, 194, 629, 536,
        580, 514, 12, 72, 385, 499, 745, 248, 742, 369, 71, 699, 348, 58, 758, 417,
        214, 120, 215, 81, 219, 288, 423, 41, 704, 522, 49, 490, 276, 589, 614, 760,
        647, 388, 64, 453, 308, 516, 433, 210, 465, 300, 425, 3, 410, 17, 286, 492,
        508, 432, 447, 193, 383, 361, 126, 128, 584, 87, 173, 23, 343, 554, 281, 654,
        392, 560, 161, 252, 146, 125, 265, 291, 561, 569, 205, 420, 436, 212, 539, 116,
        34, 283, 88, 441, 246, 540, 733, 201, 95, 309, 319, 714, 763, 147, 659, 52,
        257, 360, 30, 542, 56, 675, 115, 571, 112, 93, 619, 613, 365, 412, 221, 339,
        408, 243, 674, 535, 294, 370, 347, 604, 764, 104, 434, 648, 326, 409, 473, 211,
        565, 487, 677, 452, 313, 268, 21, 652, 558, 20, 757, 437, 180, 203, 327, 228,
        254, 184, 131, 707, 616, 65, 101, 33, 60, 310, 165, 189, 79, 170, 525, 66,
        249, 43, 134, 685, 680, 401, 36, 478, 108, 199, 109, 328, 721, 587, 570, 458,
        460, 701, 91, 741, 314, 523, 767, 330, 311, 534, 381, 234, 377, 585, 50, 506,
        307, 90, 702, 354, 279, 176, 247, 750, 230, 74, 117, 137, 267, 754, 484, 295,
        148, 200, 237, 113, 368, 679, 521, 427, 4, 737, 725, 258, 749, 722, 602, 661,
        224, 260, 708, 303, 395, 582, 466, 69, 14, 315, 672, 31, 730, 302, 355, 98,
        736, 391, 80, 298, 318, 498, 157, 509, 642, 656, 588, 612, 192, 483, 48, 645,
        734, 287, 549, 731, 358, 713, 691, 338, 99, 431, 139, 493, 59, 372, 337, 765,
        735, 261, 51, 375, 520, 511, 421, 138, 28, 82, 467, 336, 1, 689, 438, 100,
        755, 688, 97, 507, 715, 517, 692, 567, 468, 726, 744, 367, 68, 38, 156, 86,
        717, 241, 344, 405, 251, 700, 0, 444, 143, 740, 646, 573, 634, 325, 668, 524,
        598, 127, 229, 546, 274, 334, 705, 697, 556, 470, 486, 557, 476, 122, 463, 263,
        435, 548, 92, 620, 693, 579, 262, 631, 577, 289, 242, 301, 204, 207, 727, 169,
        110, 22, 676, 345, 384, 428, 197, 503, 304, 5, 239, 728, 62, 320, 11, 449,
        27, 2, 121, 510, 342, 657, 653, 78, 111, 617, 393, 240, 196, 191, 501, 282,
        24, 564, 718, 346, 627, 277, 119, 292, 159, 658, 269, 317, 761, 349, 474, 748,
        373, 443, 232, 519, 19, 226, 684, 364, 259, 537, 324, 637, 550, 600, 150, 61,
        102, 235, 179, 446, 351, 480, 669, 762, 595, 177, 581, 164, 442, 593, 481, 124,
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
