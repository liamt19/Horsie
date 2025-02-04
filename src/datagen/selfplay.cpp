

#include "selfplay.h"

#include "../movegen.h"
#include "../position.h"
#include "../threadpool.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#if defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif
#include <cstring>

namespace Horsie {

    namespace Datagen {

        inline void ClearConsoleLines(i32 nLines) {
            std::cout << "\033[H";

#if defined(_WIN64)
            CONSOLE_SCREEN_BUFFER_INFO csbi;
            GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
            int columns = csbi.srWindow.Right - csbi.srWindow.Left + 1;
#else
            struct winsize w;
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
            i32 columns = w.ws_col;
#endif

            for (size_t i = 0; i < nLines; i++) {
                std::cout << std::string(' ', columns) << "\n";
            }

            std::cout << "\033[H";
        }

        inline void HideCursor() {
#if defined(_WIN64)
            HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
            CONSOLE_CURSOR_INFO cursorInfo;
            GetConsoleCursorInfo(hConsole, &cursorInfo);
            cursorInfo.bVisible = FALSE;
            SetConsoleCursorInfo(hConsole, &cursorInfo);
#endif
        }

        std::atomic<i32> seed(static_cast<i32>(std::chrono::steady_clock::now().time_since_epoch().count()));
        
        thread_local std::mt19937 rng([]{ return std::mt19937(seed.fetch_add(1, std::memory_order_relaxed)); }());

        static i32 RandNext(i32 min, i32 max) {
            std::uniform_int_distribution<> distr(min, max - 1);
            return distr(rng);
        }


        void MonitorDG(u64 nodes, u64 depth, u64 games, u64 threads, bool dfrc) {
            gameCounts.resize(threads);
            positionCounts.resize(threads);
            threadNPS.resize(threads);

            std::vector<std::thread> dg_threads;
            for (i32 i = 0; i < threads; i++) {
                dg_threads.emplace_back(Datagen::RunGames, i, 5000, 14, games, dfrc);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            HideCursor();

            while (true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                ClearConsoleLines(threads + 2);

                u64 totalGames = 0;
                double totalNPS = 0;
                u64 totalPositions = 0;

                for (i32 i = 0; i < threads; i++) {
                    const auto games = gameCounts[i];
                    const auto positions = positionCounts[i];
                    const auto nps = threadNPS[i];
                    std::cout << std::format("Thread {:3}: {:>12} {:>15} {:>12.2f}\n", i, games, positions, nps);

                    totalGames += games;
                    totalPositions += positions;
                    totalNPS += nps;
                }
                std::cout << "           ------------------------------------------" << "\n";
                std::cout << "            " << std::format("{:>12} {:>15} {:>12.2f}", totalGames, totalPositions, totalNPS) << std::endl;
            }

            for (auto& thread : dg_threads) {
                thread.join();
            }
        }


        void RunGames(i32 threadID, u64 softNodeLimit, u64 depthLimit, u64 gamesToRun, bool dfrc) {
            Horsie::Hash = HashSize;
            Horsie::UCI_Chess960 = dfrc;
            
            const auto tt = std::make_unique<TranspositionTable>();
            const auto thread = std::make_unique<SearchThread>();

            tt->Initialize(Horsie::Hash);
            thread->TT = tt.get();
            thread->ThreadIdx = 0;
            thread->OnDepthFinish = []() {};
            thread->OnSearchFinish = []() {};

            Position pos = Position();
            Bitboard& bb = pos.bb;

            ScoredMove legalMoves[MoveListSize];
            Move bestMove = Move::Null();
            i32 bestMoveScore = 0;

            std::stringstream iss;
            iss << "";
            iss << "./data/";
            if (dfrc) iss << "dfrc_";
            iss << std::to_string(softNodeLimit / 1000) << "k_";
            iss << std::to_string(depthLimit) << "d_";
            iss << std::to_string(threadID) << ".bin";
            std::string fName = iss.str();

            std::ofstream outputWriter(fName, std::ios::binary);
#if defined(PLAINTEXT)
            std::vector<PlaintextDataFormat> datapoints{};
#else
            std::vector<BulletFormatEntry> datapoints{};
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

            std::chrono::system_clock::time_point startTime = std::chrono::system_clock::now();

            for (u64 gameNum = 0; gameNum < gamesToRun; gameNum++) {
                thread->SetStop(false);
                thread->TT->Clear();
                thread->TT->TTUpdate();
                thread->History.Clear();
                datapoints.clear();

                if (dfrc) {
                    pos.SetupForDFRC(RandNext(0, 960), RandNext(0, 960));
                }
                else {
                    pos.LoadFromFEN(InitialFEN);
                }

                int randMoveCount = RandNext(MinOpeningPly, MaxOpeningPly + 1);
                for (int i = 0; i < randMoveCount; i++) {
                    int l = Generate<GenLegal>(pos, legalMoves, 0);
                    if (l == 0) { gameNum--; continue; }
                    pos.MakeMove(legalMoves[RandNext(0, l)].move);
                }

                if (Generate<GenLegal>(pos, legalMoves, 0) == 0) { gameNum--; continue; }

                DGSetupThread(pos, *thread);
                thread->Search(prelimInfo);
                if (std::abs(thread->RootMoves[0].Score) >= MaxOpeningScore) { gameNum--; continue; }

                GameResult result = GameResult::Draw;
                int toWrite = 0;
                int filtered = 0;
                int adjudicationCounter = 0;

                while (true) {
                    DGSetupThread(pos, *thread);
                    thread->Search(info);

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
                        datapoints.emplace_back(BulletFormatEntry(pos, bestMove, bestMoveScore));
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

                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime).count();
                auto goodPerSec = totalGoodPositions / (static_cast<double>(duration) / 1000);

                gameCounts[threadID] = gameNum;
                positionCounts[threadID] = totalGoodPositions;
                threadNPS[threadID] = goodPerSec;

                AddResultsAndWrite(datapoints, result, outputWriter);
            }
        }


        void AddResultsAndWrite(std::vector<BulletFormatEntry> datapoints, GameResult gr, std::ofstream& outputWriter) {
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
            std::memset(st, 0, StateCopySize);
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


        void DGSetupThread(Position& pos, SearchThread& td) {
            td.Reset();

            ScoredMove rms[MoveListSize] = {};
            i32 size = Generate<GenLegal>(pos, &rms[0], 0);

            td.RootMoves.clear();
            td.RootMoves.shrink_to_fit();
            td.RootMoves.reserve(size);

            for (i32 j = 0; j < size; j++) {
                td.RootMoves.push_back(RootMove(rms[j].move));
            }

            td.RootPosition.LoadFromFEN(pos.GetFEN());
        }

    }
}