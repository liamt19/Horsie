#pragma once

#include <iostream>
#include <fstream>

#include <bit>
#include <span>

#include "../position.h"
#include "../bitboard.h"
#include "../move.h"


namespace Horsie {

    namespace Datagen {

        constexpr auto HashSize = 8;

        constexpr auto MinOpeningPly = 8;
        constexpr auto MaxOpeningPly = 9;

        constexpr auto SoftNodeLimit = 5000;

        constexpr auto DepthLimit = 14;

        constexpr auto WritableDataLimit = 512;

        constexpr auto AdjudicateMoves = 4;
        constexpr auto AdjudicateScore = 3000;

        constexpr auto MaxFilteringScore = 6000;
        constexpr auto MaxOpeningScore = 1200;

        constexpr auto MaxScrambledOpeningScore = 600;

        inline std::vector<u64> gameCounts{};
        inline std::vector<u64> positionCounts{};

        enum GameResult {
            WhiteWin = 2,
            Draw = 1,
            BlackWin = 0
        };

        struct BulletFormatEntry {
            u64 occ;
            u8 pcs[16];
            i16 score;
            u8 result;
            u8 ksq;
            u8 opp_ksq;
            u8 _pad[3];

            BulletFormatEntry(u64 occ, i16 score, u8 result, u8 ksq, u8 opp_ksq)
                : occ(occ), score(score), result(result), ksq(ksq), opp_ksq(opp_ksq) {
                for (i32 i = 0; i < 16; i++) {
                    pcs[i] = 0;
                }
            }

            BulletFormatEntry() {}

            void FillBitboard(Bitboard& bb) const {
                bb.Reset();

                bb.Occupancy = occ;

                u64 temp = occ;
                int idx = 0;
                while (temp != 0) {
                    int sq = poplsb(temp);
                    int piece = (pcs[idx / 2] >> (4 * (idx & 1))) & 0b1111;

                    bb.AddPiece(sq, piece / 8, piece % 8);

                    idx++;
                }
            }



            static BulletFormatEntry FromBitboard(Bitboard& bb, int stm, short score, GameResult result) {
                u64 bbs[8] = {
                    bb.Colors[WHITE], bb.Colors[BLACK],
                    bb.Pieces[0], bb.Pieces[1], bb.Pieces[2],
                    bb.Pieces[3], bb.Pieces[4], bb.Pieces[5],
                };

                if (stm == BLACK) {
                    for (auto& bb : bbs)
                        bb = std::byteswap(bb);

                    std::swap(bbs[WHITE], bbs[BLACK]);

                    score = (short)-score;
                    result = static_cast<GameResult>(1 - result);
                }

                u64 occ = bbs[0] | bbs[1];

                u8 result_ = static_cast<u8>((2 * static_cast<i32>(result)));
                u8 ksq = static_cast<u8>(std::countr_zero(bbs[0] & bbs[7]));
                u8 opp_ksq = static_cast<u8>(std::countr_zero(bbs[1] & bbs[7]) ^ 56);
                BulletFormatEntry bfe(occ, score, result_, ksq, opp_ksq);

                i32 idx = 0;
                u64 occ2 = occ;
                i32 piece = 0;
                while (occ2 > 0) {
                    i32 sq = std::countr_zero(occ2);
                    u64 bit = 1ULL << sq;
                    occ2 &= occ2 - 1;

                    u8 colour = static_cast<u8>(((bit & bbs[1]) > 0 ? 1 : 0) << 3);
                    for (int i = 2; i < 8; i++)
                        if ((bit & bbs[i]) > 0) {
                            piece = i - 2;
                            break;
                        }

                    u8 pc = static_cast<u8>(colour | static_cast<u8>(piece));

                    bfe.pcs[idx / 2] |= static_cast<u8>(pc << (4 * (idx & 1)));

                    idx += 1;
                }

                return bfe;
            }

            void WriteToBuffer(std::span<u8> buff) const {
                std::memcpy(&buff[0], reinterpret_cast<const u8*>(this), sizeof(BulletFormatEntry));
            }
        };

#pragma pack(push, 1)
        struct BulletDataFormat {
            union {
                BulletFormatEntry BFE;
                struct {
                    u8 _pad[29];
                    Move BestMove;
                    u8 STM;
                };
            };


            BulletDataFormat() {}
            BulletDataFormat(Position& pos, Move bestMove, i32 score) { Fill(pos, bestMove, score); }

            constexpr i32 Score() const { return BFE.score; }
            constexpr void SetScore(i32 v) { BFE.score = static_cast<i16>(v); }

            constexpr GameResult Result() const { return static_cast<GameResult>(BFE.result); }
            constexpr void SetResult(GameResult v) { BFE.result = static_cast<u8>(v); }

            void Fill(Position& pos, Move bestMove, i32 score) {
                BFE = BulletFormatEntry::FromBitboard(pos.bb, pos.ToMove, (short)score, GameResult::Draw);
                STM = (u8)pos.ToMove;
                BestMove = bestMove;
            }

            friend std::ofstream& operator<<(std::ofstream& os, const BulletDataFormat& data) {
                os.write(reinterpret_cast<const char*>(&data), sizeof(BulletDataFormat));
                return os;
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


        void MonitorDG();

        void RunGames(u64 gamesToRun, i32 threadID, u64 softNodeLimit = SoftNodeLimit, u64 depthLimit = DepthLimit, bool dfrc = false);
        void ResetPosition(Position& pos, CastlingStatus cr = CastlingStatus::None);

        template <bool plaintext>
        void AddResultsAndWrite(std::vector<BulletDataFormat> datapoints, GameResult gr, std::ofstream& outputWriter);

    }
}