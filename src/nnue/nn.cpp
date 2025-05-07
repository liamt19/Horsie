
#include "../nnue/nn.h"

#include "../3rdparty/zstd/zstd.h"
#include "../nnue/simd.h"

#include <cstring>
#include <fstream>
#include <memory>

#ifdef _MSC_VER
#define HIDE_MSVC
#pragma push_macro("_MSC_VER")
#undef _MSC_VER
#endif
#include "../3rdparty/incbin/incbin.h"
#ifdef HIDE_MSVC
#pragma pop_macro("_MSC_VER")
#undef HIDE_MSVC
#endif

namespace {
    INCBIN(EVAL, EVALFILE);
}

namespace Horsie::NNUE {

#if defined(COUNT_FOR_PERMUTATIONS)
    std::array<u64, L1_SIZE> NNZCounts = {};
    u64 ActivationCount = 0;
    u64 EvalCalls = 0;
#endif

    NNZTable nnzTable;
    Network net;
    const Network* g_network;

    void LoadNetwork(const std::string& path) {
        SetupNNZ();

        g_network = &net;
        std::unique_ptr<QuantisedNetwork> UQNet = std::make_unique<QuantisedNetwork>();
        const auto dst = reinterpret_cast<std::byte*>(UQNet.get());

#if defined(VS_COMP)
        std::ifstream stream(path, std::ios::binary);
#else
        std::istringstream stream(std::string(reinterpret_cast<const char*>(gEVALData), gEVALSize));
#endif

        if (IsCompressed(stream)) {
            LoadZSTD(stream, dst);
        }
        else {
            stream.read(reinterpret_cast<char*>(dst), sizeof(QuantisedNetwork));
        }

        auto& tempL1 = UQNet->L1Weights;

        for (size_t i = 0; i < INPUT_SIZE * L1_SIZE * INPUT_BUCKETS; i++)
            net.FTWeights[i] = UQNet->FTWeights[i];

        for (size_t i = 0; i < L1_SIZE; i++)
            net.FTBiases[i] = UQNet->FTBiases[i];

        PermuteFT(Span<i16>(net.FTWeights), Span<i16>(net.FTBiases));
        PermuteL1(tempL1);

        for (i32 bucket = 0; bucket < OUTPUT_BUCKETS; bucket++) {

            for (i32 i = 0; i < L1_SIZE / L1_CHUNK_PER_32; ++i)
                for (i32 j = 0; j < L2_SIZE; ++j)
                    for (i32 k = 0; k < L1_CHUNK_PER_32; ++k)
                        net.L1Weights[bucket][i * L1_CHUNK_PER_32 * L2_SIZE
                        + j * L1_CHUNK_PER_32
                        + k] = tempL1[i * L1_CHUNK_PER_32 + k][bucket][j];

            for (i32 i = 0; i < L2_SIZE; ++i)
                net.L1Biases[bucket][i] = UQNet->L1Biases[bucket][i];

            for (i32 i = 0; i < L2_SIZE; ++i)
                for (i32 j = 0; j < L3_SIZE; ++j)
                    net.L2Weights[bucket][i * L3_SIZE + j] = UQNet->L2Weights[i][bucket][j];

            for (i32 i = 0; i < L3_SIZE; ++i)
                net.L2Biases[bucket][i] = UQNet->L2Biases[bucket][i];

            for (i32 i = 0; i < L3_SIZE; ++i)
                net.L3Weights[bucket][i] = UQNet->L3Weights[i][bucket];

            net.L3Biases[bucket] = UQNet->L3Biases[bucket];
        }

        auto ws = reinterpret_cast<__m128i*>(&net.FTWeights);
        auto bs = reinterpret_cast<__m128i*>(&net.FTBiases);
        const i32 numChunks = sizeof(__m128i) / sizeof(i16);
#if defined(AVX512)
        const i32 numRegi = 8;
        constexpr i32 order[] = { 0, 2, 4, 6, 1, 3, 5, 7 };
#elif defined(AVX256)
        const i32 numRegi = 4;
        constexpr i32 order[] = { 0, 2, 1, 3 };
#else
        const i32 numRegi = 2;
        constexpr i32 order[] = { 0, 1 };
#endif
        __m128i regi[numRegi] = {};

        for (i32 i = 0; i < INPUT_SIZE * L1_SIZE * INPUT_BUCKETS / numChunks; i += numRegi) {
            for (i32 j = 0; j < numRegi; j++)
                regi[j] = ws[i + j];

            for (i32 j = 0; j < numRegi; j++)
                ws[i + j] = regi[order[j]];
        }

        for (i32 i = 0; i < L1_SIZE / numChunks; i += numRegi) {
            for (i32 j = 0; j < numRegi; j++)
                regi[j] = bs[i + j];

            for (i32 j = 0; j < numRegi; j++)
                bs[i + j] = regi[order[j]];
        }
    }

    static void SetupNNZ() {
        for (u32 i = 0; i < 256; i++) {
            u16* ptr = reinterpret_cast<u16*>(&nnzTable.Entries[i]);

            u32 j = i;
            u32 k = 0;
            while (j != 0) {
                u32 lsbIndex = std::countr_zero(j);
                j &= j - 1;
                ptr[k++] = static_cast<u16>(lsbIndex);
            }
        }
    }


    i32 GetEvaluation(Position& pos) {
        const auto occ = popcount(pos.bb.Occupancy);
        const auto outputBucket = (occ - 2) / ((32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS);

        return GetEvaluation(pos, outputBucket);
    }

    i32 GetEvaluation(Position& pos, i32 outputBucket) {
        const auto accumulator = pos.CurrAccumulator();
        pos.Accumulators.EnsureUpdated(pos);

        const auto us = Span<i16>(accumulator->Sides[pos.ToMove]);
        const auto them = Span<i16>(accumulator->Sides[Not(pos.ToMove)]);

        i8 ft_outputs[L1_SIZE];
        float L1Outputs[L2_SIZE];
        float L2Outputs[L3_SIZE];
        float L3Output = 0;

        i32 nnzCount = 0;
        u16 nnzIndices[L1_SIZE / L1_CHUNK_PER_32];

        {
            const auto zero = vec_setzero_epi16();
            const auto one = vec_set1_epi16(FT_QUANT);

            i32 offset = 0;

            const vec_128i baseInc = _mm_set1_epi16(u16(NNZ_INCREMENT));
            vec_128i baseVec = _mm_setzero_si128();

            for (const auto acc : { us, them }) {
                for (i32 i = 0; i < L1_PAIR_COUNT; i += (I16_CHUNK_SIZE * 2)) {
                    const auto input0a = vec_load_epi16(reinterpret_cast<const vec_i16*>(&acc[i + 0 * I16_CHUNK_SIZE + 0]));
                    const auto input0b = vec_load_epi16(reinterpret_cast<const vec_i16*>(&acc[i + 1 * I16_CHUNK_SIZE + 0]));

                    const auto input1a = vec_load_epi16(reinterpret_cast<const vec_i16*>(&acc[i + 0 * I16_CHUNK_SIZE + L1_PAIR_COUNT]));
                    const auto input1b = vec_load_epi16(reinterpret_cast<const vec_i16*>(&acc[i + 1 * I16_CHUNK_SIZE + L1_PAIR_COUNT]));

                    const auto clipped0a = vec_min_epi16(vec_max_epi16(input0a, zero), one);
                    const auto clipped0b = vec_min_epi16(vec_max_epi16(input0b, zero), one);

                    const auto clipped1a = vec_min_epi16(input1a, one);
                    const auto clipped1b = vec_min_epi16(input1b, one);

                    const auto producta = vec_mulhi_epi16(vec_slli_epi16(clipped0a, 16 - FT_SHIFT), clipped1a);
                    const auto productb = vec_mulhi_epi16(vec_slli_epi16(clipped0b, 16 - FT_SHIFT), clipped1b);

                    const auto prod = vec_packus_epi16(producta, productb);
                    vec_storeu_epi8(reinterpret_cast<vec_i8*>(&ft_outputs[offset + i]), prod);

                    const auto nnz_mask = vec_nnz_mask(prod);

                    for (i32 j = 0; j < NNZ_OUTPUTS_PER_CHUNK; j++) {
                        i32 lookup = (nnz_mask >> (j * 8)) & 0xFF;
                        auto offsets = nnzTable.Entries[lookup];
                        _mm_storeu_si128(reinterpret_cast<vec_128i*>(&nnzIndices[nnzCount]), _mm_add_epi16(baseVec, offsets));

                        nnzCount += std::popcount(static_cast<u32>(lookup));
                        baseVec = _mm_add_epi16(baseVec, baseInc);
                    }
                }

                offset += L1_PAIR_COUNT;
            }
        }


        {
            const auto inputs = Span<i8>(ft_outputs);
            const auto& weights = net.L1Weights[outputBucket];
            const auto& biases = net.L1Biases[outputBucket];
            const auto outputs = L1Outputs;

            vec_i32 sums[L2_SIZE / I32_CHUNK_SIZE]{};

            const auto inputs32 = (i32*)(inputs.data());
            for (i32 i = 0; i < nnzCount; i++) {
                const auto index = nnzIndices[i];
                const auto input32 = vec_set1_epi32(inputs32[index]);
                const auto weight = reinterpret_cast<const vec_i8*>(&weights[index * L1_CHUNK_PER_32 * L2_SIZE]);
                for (i32 k = 0; k < L2_SIZE / F32_CHUNK_SIZE; k++)
                    sums[k] = vec_dpbusd_epi32(sums[k], input32, weight[k]);
            }

            const auto sumMul = vec_set1_ps(L1_MUL);

            const auto zero = vec_set1_ps(0.0f);
            const auto one = vec_set1_ps(1.0f);
            for (i32 i = 0; i < L2_SIZE / F32_CHUNK_SIZE; ++i) {
                const auto biasVec = vec_loadu_ps(&biases[i * F32_CHUNK_SIZE]);
                const auto sumPs = vec_fmadd_ps(vec_cvtepi32_ps(sums[i]), sumMul, biasVec);
                const auto clipped = vec_min_ps(vec_max_ps(sumPs, zero), one);
                const auto squared = vec_mul_ps(clipped, clipped);
                vec_storeu_ps(&outputs[i * F32_CHUNK_SIZE], squared);
            }
        }


        {
            const auto inputs = L1Outputs;
            const auto& weights = net.L2Weights[outputBucket];
            const auto& biases = net.L2Biases[outputBucket];
            const auto outputs = L2Outputs;

            vec_ps sumVecs[L3_SIZE / F32_CHUNK_SIZE];

            for (i32 i = 0; i < L3_SIZE / F32_CHUNK_SIZE; ++i)
                sumVecs[i] = vec_loadu_ps(&biases[i * F32_CHUNK_SIZE]);

            for (i32 i = 0; i < L2_SIZE; ++i) {
                const auto inputVec = vec_set1_ps(inputs[i]);
                const auto weight = reinterpret_cast<const vec_ps*>(&weights[i * L3_SIZE]);
                for (i32 j = 0; j < L3_SIZE / F32_CHUNK_SIZE; ++j)
                    sumVecs[j] = vec_fmadd_ps(inputVec, weight[j], sumVecs[j]);
            }

            const auto zero = vec_set1_ps(0.0f);
            const auto one = vec_set1_ps(1.0f);
            for (i32 i = 0; i < L3_SIZE / F32_CHUNK_SIZE; ++i) {
                const auto clipped = vec_min_ps(vec_max_ps(sumVecs[i], zero), one);
                const auto squared = vec_mul_ps(clipped, clipped);
                vec_storeu_ps(&outputs[i * F32_CHUNK_SIZE], squared);
            }
        }


        {
            const auto& inputs = L2Outputs;
            const auto& weights = net.L3Weights[outputBucket];
            const auto bias = net.L3Biases[outputBucket];

            constexpr auto SUM_COUNT = 64 / sizeof(vec_ps);
            vec_ps sumVecs[SUM_COUNT]{};

            for (i32 i = 0; i < L3_SIZE / F32_CHUNK_SIZE; i++) {
                const auto weightVec = vec_loadu_ps(&weights[i * F32_CHUNK_SIZE]);
                const auto inputsVec = vec_loadu_ps(&inputs[i * F32_CHUNK_SIZE]);
                sumVecs[i % SUM_COUNT] = vec_fmadd_ps(inputsVec, weightVec, sumVecs[i % SUM_COUNT]);
            }

            L3Output = bias + vec_hsum_ps(sumVecs);
        }

        return static_cast<i32>(L3Output * OutputScale);
    }


    std::pair<i32, i32> FeatureIndex(i32 pc, i32 pt, i32 sq, i32 wk, i32 bk) {
        const i32 ColorStride = 64 * 6;
        const i32 PieceStride = 64;

        i32 wSq = sq;
        i32 bSq = sq ^ 56;

        if (wk % 8 > 3) {
            wk ^= 7;
            wSq ^= 7;
        }

        bk ^= 56;
        if (bk % 8 > 3) {
            bk ^= 7;
            bSq ^= 7;
        }

        i32 whiteIndex = (768 * KingBuckets[wk]) + (pc * ColorStride) + (pt * PieceStride) + wSq;
        i32 blackIndex = (768 * KingBuckets[bk]) + (Not(pc) * ColorStride) + (pt * PieceStride) + bSq;

        return { whiteIndex * L1_SIZE, blackIndex * L1_SIZE };
    }

    i32 FeatureIndexSingle(i32 pc, i32 pt, i32 sq, i32 kingSq, i32 perspective) {
        const i32 ColorStride = 64 * 6;
        const i32 PieceStride = 64;

        if (perspective == BLACK) {
            sq ^= 56;
            kingSq ^= 56;
        }

        if (kingSq % 8 > 3) {
            sq ^= 7;
            kingSq ^= 7;
        }

        return ((768 * KingBuckets[kingSq]) + ((pc ^ perspective) * ColorStride) + (pt * PieceStride) + (sq)) * L1_SIZE;
    }


    void ResetCaches(Position& pos) {
        for (auto& bucket : pos.CachedBuckets) {
            bucket.accumulator.Sides[WHITE] = bucket.accumulator.Sides[BLACK] = g_network->FTBiases;
            bucket.Boards[WHITE].Reset();
            bucket.Boards[BLACK].Reset();
        }
    }

    //  See https://github.com/Ciekce/Stormphrax/pull/176 for the non-scuffed impl of this.
    void LoadZSTD(std::istream& m_stream, std::byte* dst) {
        size_t m_pos = 0;
        size_t m_end = 0;
        size_t m_result = 0;

        std::vector<std::byte> m_inBuf(ZSTD_DStreamInSize());
        std::vector<std::byte> m_outBuf(ZSTD_DStreamOutSize());
        ZSTD_inBuffer m_input{};
        ZSTD_outBuffer m_output{};

        m_input.src = m_inBuf.data();
        m_output.dst = m_outBuf.data();

        auto m_dStream = ZSTD_createDStream();

        size_t n = sizeof(QuantisedNetwork);
        while (n > 0) {
            while (m_pos >= m_end && (m_input.pos < m_input.size || !m_stream.fail())) {

                if (m_input.pos == m_input.size) {
                    if (!m_stream) {
                        break;
                    }

                    m_stream.read(reinterpret_cast<char*>(m_inBuf.data()), static_cast<std::streamsize>(m_inBuf.size()));

                    m_input.size = m_stream.gcount();
                    m_input.pos = 0;
                }

                if (m_result == 0 && m_input.pos < m_input.size)
                    ZSTD_initDStream(m_dStream);

                m_output.size = m_outBuf.size();
                m_output.pos = 0;

                m_result = ZSTD_decompressStream(m_dStream, &m_output, &m_input);

                if (ZSTD_isError(m_result)) {
                    const auto code = ZSTD_getErrorCode(m_result);
                    std::cerr << "zstd error: " << ZSTD_getErrorString(code) << std::endl;
                    break;
                }

                m_pos = 0;
                m_end = m_output.pos;
            }

            if (m_pos >= m_end) {
                ZSTD_freeDStream(m_dStream);
                return;
            }

            const auto remaining = m_end - m_pos;
            const auto count = std::min(n, remaining);

            std::memcpy(dst, &m_outBuf[m_pos], count);

            m_pos += count;
            dst += count;

            n -= count;
        }

        ZSTD_freeDStream(m_dStream);
    }

    bool IsCompressed(std::istream& stream) {
        const i32 ZSTD_HEADER = -47205080;
        const i32 headerMaybe = read_little_endian<i32>(stream);
        stream.seekg(0);

        return (headerMaybe == ZSTD_HEADER);
    }

    static void PermuteFT(Span<i16> ftWeights, Span<i16> ftBiases) {
        const i32 OneBucket = (INPUT_SIZE * L1_SIZE);
        std::vector<i16> temp(OneBucket, 0);

        for (i32 bucket = 0; bucket < INPUT_BUCKETS; bucket++) {
            Span<i16> ftBucket = Span<i16>(&ftWeights[(bucket * OneBucket)], OneBucket);

            for (size_t i = 0; i < OneBucket; i++)
                temp[i] = ftBucket[i];

            for (i32 i = 0; i < INPUT_SIZE; i++) {
                for (i32 dst = 0; dst < L1_PAIR_COUNT; dst++) {
                    i32 src = PermuteIndices[dst];
                    auto f = i * L1_SIZE;

                    ftBucket[f + dst] = temp[f + src];
                    ftBucket[f + dst + L1_PAIR_COUNT] = temp[f + src + L1_PAIR_COUNT];
                }
            }
        }

        std::vector<i16> temp2(L1_SIZE, 0);
        for (i32 i = 0; i < L1_SIZE; i++)
            temp2[i] = ftBiases[i];

        for (i32 dst = 0; dst < L1_PAIR_COUNT; dst++) {
            i32 src = PermuteIndices[dst];

            ftBiases[dst] = temp2[src];
            ftBiases[dst + L1_PAIR_COUNT] = temp2[src + L1_PAIR_COUNT];
        }
    }

    static void PermuteL1(i8 l1Weights[L1_SIZE][OUTPUT_BUCKETS][L2_SIZE]) {
        auto temp = new i8[L1_SIZE][OUTPUT_BUCKETS][L2_SIZE];
        std::memcpy(temp, l1Weights, N_L1W);

        for (i32 dst = 0; dst < L1_PAIR_COUNT; dst++) {
            i32 src = PermuteIndices[dst];

            for (i32 b = 0; b < OUTPUT_BUCKETS; b++) {
                for (i32 l2 = 0; l2 < L2_SIZE; l2++) {
                    l1Weights[dst][b][l2] = temp[src][b][l2];
                    l1Weights[dst + L1_PAIR_COUNT][b][l2] = temp[src + L1_PAIR_COUNT][b][l2];
                }
            }
        }

        delete[] temp;
    }

    void SubSubAddAdd(const i16* _src, i16* _dst, const i16* _sub1, const i16* _sub2, const i16* _add1, const i16* _add2) {
        const vec_i16* src = reinterpret_cast<const vec_i16*>(_src);
        vec_i16* dst = reinterpret_cast<vec_i16*>(_dst);
        const vec_i16* sub1 = reinterpret_cast<const vec_i16*>(_sub1);
        const vec_i16* sub2 = reinterpret_cast<const vec_i16*>(_sub2);
        const vec_i16* add1 = reinterpret_cast<const vec_i16*>(_add1);
        const vec_i16* add2 = reinterpret_cast<const vec_i16*>(_add2);
        for (i32 i = 0; i < SIMD_CHUNKS; i++) {
            dst[i] = vec_sub_epi16(vec_sub_epi16(vec_add_epi16(vec_add_epi16(src[i], add1[i]), add2[i]), sub1[i]), sub2[i]);
        }
    }

    void SubSubAdd(const i16* _src, i16* _dst, const i16* _sub1, const i16* _sub2, const i16* _add1) {
        const vec_i16* src = reinterpret_cast<const vec_i16*>(_src);
        vec_i16* dst = reinterpret_cast<vec_i16*>(_dst);
        const vec_i16* sub1 = reinterpret_cast<const vec_i16*>(_sub1);
        const vec_i16* sub2 = reinterpret_cast<const vec_i16*>(_sub2);
        const vec_i16* add1 = reinterpret_cast<const vec_i16*>(_add1);
        for (i32 i = 0; i < SIMD_CHUNKS; i++) {
            dst[i] = vec_sub_epi16(vec_sub_epi16(vec_add_epi16(src[i], add1[i]), sub1[i]), sub2[i]);
        }
    }

    void SubAdd(const i16* _src, i16* _dst, const i16* _sub1, const i16* _add1) {
        const vec_i16* src = reinterpret_cast<const vec_i16*>(_src);
        vec_i16* dst = reinterpret_cast<vec_i16*>(_dst);
        const vec_i16* sub1 = reinterpret_cast<const vec_i16*>(_sub1);
        const vec_i16* add1 = reinterpret_cast<const vec_i16*>(_add1);
        for (i32 i = 0; i < SIMD_CHUNKS; i++) {
            dst[i] = vec_sub_epi16(vec_add_epi16(src[i], add1[i]), sub1[i]);
        }
    }

    void Add(const i16* _src, i16* _dst, const i16* _add1) {
        const vec_i16* src = reinterpret_cast<const vec_i16*>(_src);
        vec_i16* dst = reinterpret_cast<vec_i16*>(_dst);
        const vec_i16* add1 = reinterpret_cast<const vec_i16*>(_add1);
        for (i32 i = 0; i < SIMD_CHUNKS; i++)
            dst[i] = vec_add_epi16(src[i], add1[i]);
    }

    void Sub(const i16* _src, i16* _dst, const i16* _sub1) {
        const vec_i16* src = reinterpret_cast<const vec_i16*>(_src);
        vec_i16* dst = reinterpret_cast<vec_i16*>(_dst);
        const vec_i16* sub1 = reinterpret_cast<const vec_i16*>(_sub1);
        for (i32 i = 0; i < SIMD_CHUNKS; i++)
            dst[i] = vec_sub_epi16(src[i], sub1[i]);
    }
}
