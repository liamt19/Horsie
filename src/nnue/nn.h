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
            std::array<short, FeatureWeightElements * InputBuckets>             FeatureWeights;
            std::array<short, FeatureBiasElements>                              FeatureBiases;
            std::array<std::array<short, LayerWeightElements>, OutputBuckets>   LayerWeights;
            std::array<short, LayerBiasElements>                                LayerBiases;
        };

        extern Network net;

        void LoadNetwork(const std::string& name);

        void RefreshAccumulator(Position& pos);
        void RefreshAccumulatorPerspectiveFull(Position& pos, int perspective);
        void RefreshAccumulatorPerspective(Position& pos, int perspective);

        void MakeMoveNN(Position& pos, Move m);
        void MakeNullMove(Position& pos);
        void UpdateSingle(Accumulator* prev, Accumulator* curr, int perspective);
        void ProcessUpdates(Position& pos);

        int GetEvaluation(Position& pos);

        std::pair<int, int> FeatureIndex(int pc, int pt, int sq, int wk, int bk);
        int FeatureIndexSingle(int pc, int pt, int sq, int kingSq, int perspective);


        constexpr int KingBuckets[] = {
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
        static void Sub(const short* src, short* dst, const short* sub1);
        static void Add(const short* src, short* dst, const short* add1);
        static void SubAdd(const short* src, short* dst, const short* sub1, const short* add1);
        static void SubSubAdd(const short* src, short* dst, const short* sub1, const short* sub2, const short* add1);
        static void SubSubAddAdd(const short* src, short* dst, const short* sub1, const short* sub2, const short* add1, const short* add2);


    }

}

#endif
