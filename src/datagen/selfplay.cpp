

#include "selfplay.h"

#include "../position.h"
#include "../threadpool.h"
#include "../movegen.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <conio.h>

namespace Horsie {

    namespace Datagen {

        
        void MonitorDG() {
            const int N = 18;
            gameCounts.resize(N);
            positionCounts.resize(N);

            std::vector<std::thread> dg_threads;
            for (i32 i = 0; i < N; i++) {
                dg_threads.emplace_back(Datagen::RunGames, 100, i, 5000, 14, false);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();
            

            while (true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                system("cls");
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count();
                
                u64 totPositions = 0;
                for (i32 i = 0; i < N; i++) {
                    std::cout << "#" << i << ":\t" << gameCounts[i] << "\t" << positionCounts[i] << std::endl;
                    totPositions += positionCounts[i];
                }

                const auto pps = (totPositions / (static_cast<double>(duration) / 1000));
                std::cout << "\n" << pps << " /sec" << std::endl;
            }


            for (auto& thread : dg_threads) {
                thread.join();
            }
            std::cout << "All threads joined\n";
        }

        static i32 RandNext(std::mt19937 gen, i32 min, i32 max) {
            std::uniform_int_distribution<> distr(min, max);
            return distr(gen);
        }

        void RunGames(u64 gamesToRun, i32 threadID, u64 softNodeLimit, u64 depthLimit, bool dfrc) {
            Horsie::Hash = HashSize;
            Horsie::UCI_Chess960 = dfrc;

            const auto pool = std::make_unique<SearchThreadPool>(1);
            Position pos = Position();
            Bitboard& bb = pos.bb;
            SearchThread* thread = pool->MainThread();

            thread->OnDepthFinish = []() {};
            thread->OnSearchFinish = []() {};


            std::random_device rd;
            std::mt19937 gen(rd());

            ScoredMove legalMoves[MoveListSize];
            Move bestMove = Move::Null();
            int bestMoveScore = 0;

            std::stringstream iss;
            iss << "C:\\Programming\\Horsie\\x64\\Release\\";
            if (dfrc) iss << "dfrc_";
            iss << std::to_string(softNodeLimit / 1000) << "k_";
            iss << std::to_string(depthLimit) << "d_";
            iss << std::to_string(threadID) << ".bin";
            std::string fName = iss.str();

            std::ofstream outputWriter(fName, std::ios::binary);
#if defined(PLAINTEXT)
            std::vector<PlaintextDataFormat> datapoints{};
#else
            std::vector<BulletDataFormat> datapoints{};
#endif
            datapoints.reserve(WritableDataLimit);

            u64 totalBadPositions = 0;
            u64 totalGoodPositions = 0;

            SearchLimits info{};
            info.SoftNodeLimit = softNodeLimit;
            info.MaxNodes = softNodeLimit * 20;
            info.MaxDepth = (int)depthLimit;

            SearchLimits prelimInfo{};
            prelimInfo.SoftNodeLimit = softNodeLimit * 20;
            prelimInfo.MaxNodes = softNodeLimit * 400;
            prelimInfo.MaxDepth = std::clamp((int)depthLimit, 8, 10);

            std::chrono::system_clock::time_point startTime{};


            for (u64 gameNum = 0; gameNum < gamesToRun; gameNum++) {
                pool->TTable.Clear();
                pool->Clear();

                if (dfrc) {
                    pos.SetupForDFRC(RandNext(gen, 0, 960), RandNext(gen, 0, 960));
                    ResetPosition(pos);
                }
                else {
                    pos.LoadFromFEN(InitialFEN);
                }


                int randMoveCount = RandNext(gen, MinOpeningPly, MaxOpeningPly);
                for (int i = 0; i < randMoveCount; i++) {
                    int l = Generate<GenLegal>(pos, legalMoves, 0);
                    if (l == 0) { gameNum--; continue; }
                    pos.MakeMove(legalMoves[RandNext(gen, 0, l - 1)].move);
                }

                if (Generate<GenLegal>(pos, legalMoves, 0) == 0) { gameNum--; continue; }

                //  Check if the starting position has a reasonable score, and scrap it if it doesn't
                pool->StartSearch(pos, prelimInfo);
                pool->WaitForMain();
                if (std::abs(thread->RootMoves[0].Score) >= MaxOpeningScore) { gameNum--; continue; }

                GameResult result = GameResult::Draw;
                int toWrite = 0;
                int filtered = 0;
                int adjudicationCounter = 0;

                while (true) {
                    pool->StartSearch(pos, info);
                    pool->WaitForMain();

                    bestMove = thread->RootMoves[0].move;
                    bestMoveScore = thread->RootMoves[0].Score;
                    bestMoveScore *= (pos.ToMove == BLACK ? -1 : 1);


                    if (std::abs(bestMoveScore) >= AdjudicateScore) {
                        if (++adjudicationCounter > AdjudicateMoves) {
                            result = (bestMoveScore > 0) ? GameResult::WhiteWin : GameResult::BlackWin;
                            break;
                        }
                    }
                    else
                        adjudicationCounter = 0;


                    bool inCheck = pos.Checked();
                    bool bmCap = ((bb.GetPieceAtIndex(bestMove.To()) != NONE && !bestMove.IsCastle()) || bestMove.IsEnPassant());
                    bool badScore = std::abs(bestMoveScore) > MaxFilteringScore;
                    if (!(inCheck || bmCap || badScore)) {
                        datapoints.emplace_back(BulletDataFormat(pos, bestMove, bestMoveScore));
                        toWrite++;
                    }
                    else {
                        filtered++;
                    }

                    pos.MakeMove(bestMove);

                    if (Generate<GenLegal>(pos, legalMoves, 0) == 0) {
                        result = (pos.ToMove == WHITE) ? GameResult::BlackWin : GameResult::WhiteWin;
                        break;
                    }
                    else if (pos.IsDraw()) {
                        result = GameResult::Draw;
                        break;
                    }
                    else if (toWrite == WritableDataLimit - 1) {
                        result = bestMoveScore >  400 ? GameResult::WhiteWin :
                                 bestMoveScore < -400 ? GameResult::BlackWin :
                                                        GameResult::Draw;
                        break;
                    }
                }


                totalBadPositions += (u32)filtered;
                totalGoodPositions += (u32)toWrite;

                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime);
                auto goodPerSec = totalGoodPositions / (duration.count() * 1000);
                auto totalPerSec = (totalGoodPositions + totalBadPositions) / (duration.count() * 1000);

                //ProgressBroker.ReportProgress(threadID, gameNum, totalGoodPositions, goodPerSec);
                positionCounts[threadID] = totalGoodPositions;

                AddResultsAndWrite<false>(datapoints, result, outputWriter);
            }
        }

        template <bool plaintext>
        void AddResultsAndWrite(std::vector<BulletDataFormat> datapoints, GameResult gr, std::ofstream& outputWriter) {

            for (int i = 0; i < datapoints.size(); i++) {
                datapoints[i].SetResult(gr);
                outputWriter << datapoints[i];
            }

            outputWriter.flush();
        }

        void ResetPosition(Position& pos, CastlingStatus cr) {
            Bitboard& bb = pos.bb;

            pos.FullMoves = 1;
            pos.GamePly = 0;

            pos.State = pos.StartingState();

            auto st = pos.State;
            MemClear(st, 0, sizeof(StateInfo));
            st->CastleStatus = cr;
            st->HalfmoveClock = 0;
            st->PliesFromNull = 0;
            st->EPSquare = EP_NONE;
            st->CapturedPiece = NONE;
            st->KingSquares[WHITE] = bb.KingIndex(WHITE);
            st->KingSquares[BLACK] = bb.KingIndex(BLACK);

            pos.SetState();

            NNUE::RefreshAccumulator(pos);
        }

    }
}