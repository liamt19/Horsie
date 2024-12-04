#pragma once

#ifndef NN_H
#define NN_H

#include "../position.h"
#include "../nnue/arch.h"

#include <array>
#include <immintrin.h>



namespace Horsie
{
    namespace NNUE {

        struct alignas(64) Network
        {
            std::array<i16, FeatureWeightElements * InputBuckets>             FeatureWeights;
            std::array<i16, FeatureBiasElements>                              FeatureBiases;
            std::array<std::array<i16, LayerWeightElements>, OutputBuckets>   LayerWeights;
            std::array<i16, LayerBiasElements>                                LayerBiases;
        };

        extern Network net;

        void LoadNetwork(const std::string& name);

        void RefreshAccumulator(Position& pos);
        void RefreshAccumulatorPerspectiveFull(Position& pos, i32 perspective);
        void RefreshAccumulatorPerspective(Position& pos, i32 perspective);

        void MakeMoveNN(Position& pos, Move m);
        void MakeNullMove(Position& pos);
        void UpdateSingle(Accumulator* prev, Accumulator* curr, i32 perspective);
        void ProcessUpdates(Position& pos);

        i32 GetEvaluation(Position& pos);

        std::pair<i32, i32> FeatureIndex(i32 pc, i32 pt, i32 sq, i32 wk, i32 bk);
        i32 FeatureIndexSingle(i32 pc, i32 pt, i32 sq, i32 kingSq, i32 perspective);


        constexpr i32 KingBuckets[] = {
             0,  1,  2,  3, 19, 18, 17, 16,
             4,  5,  6,  7, 23, 22, 21, 20,
             8,  9, 10, 11, 27, 26, 25, 24,
             8,  9, 10, 11, 27, 26, 25, 24,
            12, 12, 13, 13, 29, 29, 28, 28,
            12, 12, 13, 13, 29, 29, 28, 28,
            14, 14, 15, 15, 31, 31, 30, 30,
            14, 14, 15, 15, 31, 31, 30, 30,
        };


        template<typename IntType>
        inline IntType read_little_endian(std::istream& stream) {
            IntType result;
            stream.read(reinterpret_cast<char*>(&result), sizeof(IntType));
            return result;
        }

        void ResetCaches(Position& pos);
        static int32_t hsum_8x32(__m256i v);
        static void Sub(const i16* src, i16* dst, const i16* sub1);
        static void Add(const i16* src, i16* dst, const i16* add1);
        static void SubAdd(const i16* src, i16* dst, const i16* sub1, const i16* add1);
        static void SubSubAdd(const i16* src, i16* dst, const i16* sub1, const i16* sub2, const i16* add1);
        static void SubSubAddAdd(const i16* src, i16* dst, const i16* sub1, const i16* sub2, const i16* add1, const i16* add2);


    }

}

#endif
