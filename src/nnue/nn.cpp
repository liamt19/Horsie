
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

#if defined(PERM_COUNT)
    std::array<u64, L1_SIZE> NNZCounts = {};
    u64 ActivationCount = 0;
    u64 EvalCalls = 0;
#endif


    Network net;
    const Conv2D<IN_CHANNELS, OUT_CHANNELS> conv1{ &net.Conv1Weights[0][0][0][0], &net.Conv1Biases[0] };
    const Conv2D<OUT_CHANNELS, SQUARE_NB>         conv2{ &net.Conv2Weights[0][0][0][0], &net.Conv2Biases[0] };
    const Linear<FC1_INPUTS, FC_LAYER_SIZE, false>  fc1{ &net.FC1Weights[0][0],         &net.FC1Biases[0] };
    const Linear<FC_LAYER_SIZE, 1, true>            fc2{ &net.FC2Weights[0],            &net.FC2Biases[0] };


    void LoadNetwork(const std::string& path) {

#if defined(VS_COMP)
        std::ifstream stream(path, std::ios::binary);
#else
        std::istringstream stream(std::string(reinterpret_cast<const char*>(gEVALData), gEVALSize));
#endif

        const auto dst = reinterpret_cast<std::byte*>(&net);

        if (IsCompressed(stream)) {
            LoadZSTD(stream, dst);
        }
        else {
            stream.read(reinterpret_cast<char*>(dst), sizeof(Network));
        }

    }

    static void EmplaceFeatures(const Position& pos, std::array<float, 768>& features) {
        const Bitboard& bb = pos.bb;

        const auto stm = pos.ToMove;
        const auto ntm = Not(pos.ToMove);
        const auto flip = (stm == BLACK) ? 56 : 0;

        for (int pt = PAWN; pt <= KING; pt++) {
            auto boys = bb.Pieces[pt] & bb.Colors[stm];
            auto opps = bb.Pieces[pt] & bb.Colors[ntm];

            while (boys != 0) {
                auto sq = poplsb(boys);
                auto idx = (64 * pt) + (sq ^ flip);
                features[idx] = 1.0f;
            }

            while (opps != 0) {
                int sq = poplsb(opps);
                auto idx = 384 + (64 * pt) + (sq ^ flip);
                features[idx] = 1.0f;
            }

        }
    }


    i32 GetEvaluation(const Position& pos) {

        std::array<float, 768> input768 = {};
        EmplaceFeatures(pos, input768);

        float c1_out[32 * 8 * 8];
        float c2_out[64 * 8 * 8];
        float fc1_out[128];
        float fc2_out[1];

        conv1.forward(&input768[0], c1_out);
        conv2.forward(c1_out, c2_out);
        fc1.forward(c2_out, fc1_out);
        fc2.forward(fc1_out, fc2_out);

        float output = fc2_out[0];
        output *= OutputScale;
        
        return static_cast<i32>(output);
    }


    void ResetCaches(Position& pos) {
#if defined(NO)
        for (auto& bucket : pos.CachedBuckets) {
            bucket.accumulator.Sides[WHITE] = bucket.accumulator.Sides[BLACK] = net.FTBiases;
            bucket.Boards[WHITE].Reset();
            bucket.Boards[BLACK].Reset();
        }
#endif
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

        size_t n = sizeof(Network);
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

}
