#pragma once

#include "../bitboard.h"
#include "../move.h"
#include "../nnue/nn.h"
#include "../position.h"
#include "../threadpool.h"
#include "bullet_format.h"

#include <bit>
#include <fstream>
#include <iostream>
#include <span>

namespace Horsie::Datagen {

    enum GameResult {
        BlackWin = 0,
        Draw = 1,
        WhiteWin = 2,
    };

#pragma pack(push,1)
    struct BulletFormatEntry {
        u64 _occ;
        u8 _pcs[16];
        i16 _score;
        u8 _result;
        u8 _ksq;
        u8 _opp_ksq;
        Move _move;
        u8 _stm;

        BulletFormatEntry(Position& pos, Move bestMove, i32 score) {
            Fill(pos, bestMove, score);
        }

        BulletFormatEntry(Position& pos, ScoredMove bestMove) {
            Fill(pos, bestMove.move, bestMove.score);
        }

        BulletFormatEntry() = default;

        constexpr i32 Score() const { return _score; }
        constexpr void SetScore(i32 v) { _score = static_cast<i16>(v); }

        constexpr GameResult Result() const { return static_cast<GameResult>(_result); }
        constexpr void SetResult(GameResult v) { _result = static_cast<u8>(v); }

        void Fill(Position& pos, Move bestMove, i32 score) {
            LoadFromBitboard(pos.bb, pos.ToMove, static_cast<i16>(score), GameResult::Draw);
            _move = bestMove;
            _stm = static_cast<u8>(pos.ToMove);
        }

        friend std::ofstream& operator<<(std::ofstream& os, const BulletFormatEntry& data) {
            os.write(reinterpret_cast<const char*>(&data), sizeof(BulletFormatEntry));
            return os;
        }

        void FillBitboard(Bitboard& bb) const {
            bb.Reset();

            bb.Occupancy = _occ;

            u64 temp = _occ;
            i32 idx = 0;
            while (temp != 0) {
                i32 sq = poplsb(temp);
                i32 piece = (_pcs[idx / 2] >> (4 * (idx & 1))) & 0b1111;

                bb.AddPiece(sq, piece / 8, piece % 8);

                idx++;
            }
        }

        void LoadFromBitboard(Bitboard& bb, i32 stm, i16 score, GameResult result) {
            u64 bbs[8] = {
                bb.Colors[WHITE], bb.Colors[BLACK],
                bb.Pieces[0], bb.Pieces[1], bb.Pieces[2],
                bb.Pieces[3], bb.Pieces[4], bb.Pieces[5],
            };

            if (stm == BLACK) {
                for (auto& bb : bbs)
                    bb = std::byteswap(bb);

                std::swap(bbs[WHITE], bbs[BLACK]);

                score *= -1;
                result = static_cast<GameResult>(1 - result);
            }

            u64 occ = bbs[0] | bbs[1];

            _occ = occ;
            for (i32 i = 0; i < 16; i++) {
                _pcs[i] = 0;
            }
            _score = score;
            _result = static_cast<u8>((2 * static_cast<i32>(result)));
            _ksq = static_cast<u8>(std::countr_zero(bbs[0] & bbs[7]));
            _opp_ksq = static_cast<u8>(std::countr_zero(bbs[1] & bbs[7]) ^ 56);


            i32 idx = 0;
            u64 occ2 = occ;
            i32 piece = 0;
            while (occ2 > 0) {
                i32 sq = std::countr_zero(occ2);
                u64 bit = 1ULL << sq;
                occ2 &= occ2 - 1;

                u8 color = static_cast<u8>(((bit & bbs[1]) > 0 ? 1 : 0) << 3);
                for (i32 i = 2; i < 8; i++)
                    if ((bit & bbs[i]) > 0) {
                        piece = i - 2;
                        break;
                    }

                u8 pc = static_cast<u8>(color | static_cast<u8>(piece));

                _pcs[idx / 2] |= static_cast<u8>(pc << (4 * (idx & 1)));

                idx += 1;
            }
        }

        static BulletFormatEntry FromBitboard(Bitboard& bb, i32 stm, i16 score, GameResult result) {
            BulletFormatEntry bfe{};
            bfe.LoadFromBitboard(bb, stm, score, result);
            return bfe;
        }

        void WriteToBuffer(std::span<u8> buff) const {
            std::memcpy(&buff[0], reinterpret_cast<const u8*>(this), sizeof(BulletFormatEntry));
        }
    };
#pragma pack(pop)

    struct PlaintextFormat {
        char FEN[92]{};
        i32 score{};
        GameResult result{};

        PlaintextFormat() {}
        PlaintextFormat(Position& pos, Move bestMove, i32 score) { Fill(pos, bestMove, score); }

        constexpr i32 Score() const { return score; }
        constexpr void SetScore(i32 v) { score = static_cast<i32>(v); }

        constexpr GameResult Result() const { return static_cast<GameResult>(result); }
        constexpr void SetResult(GameResult v) { result = v; }

        void Fill(Position& pos, Move bestMove, i32 score) {
            std::memset(FEN, '\0', sizeof(char) * 92);
                
            std::string fen = pos.GetFEN();
            std::copy(fen.begin(), fen.end(), FEN);
            FEN[fen.length() + 1] = '\0';
        }

        friend std::ofstream& operator<<(std::ofstream& os, const PlaintextFormat& data) {
                
            os << std::string(data.FEN) << " | " << data.score << " | " << data.result;
            return os;
        }
    };


}