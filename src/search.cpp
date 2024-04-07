
#define NO_PV_LEN 1

#include "search.h"
#include "position.h"
#include "nn.h"
#include "tt.h"
#include "movegen.h"
#include "precomputed.h"

#include <chrono>

#include "search_options.h"

using namespace Horsie::Search;

namespace Horsie {

    constexpr int MaxSearchStackPly = 256 - 10;


    constexpr int BoundLower = (int)TTNodeType::Alpha;
    constexpr int BoundUpper = (int)TTNodeType::Beta;


    void SearchThread::Search(Position& pos, SearchLimits& info) {

        MaxNodes = info.MaxNodes;
        SearchTimeMS = info.MaxSearchTime;

        History.Clear();

        SearchStackEntry _SearchStackBlock[MaxPly] = {};
        SearchStackEntry* ss = &_SearchStackBlock[10];
        for (int i = -10; i < MaxSearchStackPly; i++)
        {
            (ss + i)->Clear();
            (ss + i)->Ply = (short)i;
            (ss + i)->PV = (Move*)AlignedAllocZeroed((MaxPly * sizeof(Move)), AllocAlignment);
            (ss + i)->PVLength = 0;
            (ss + i)->ContinuationHistory = &History.Continuations[0][0].Histories[0][0];
        }

        for (int sq = 0; sq < SQUARE_NB; sq++)
        {
            //Array.Clear(NodeTable[sq]);
            std::memset(NodeTable[sq], 0, sizeof(NodeTable[sq]));
        }


        RootMoves.clear();
        ScoredMove rms[MoveListSize] = {};
        int rmsSize = Generate<GenLegal>(pos, &rms[0], 0);
        for (size_t i = 0; i < rmsSize; i++)
        {
            RootMove rm = RootMove(rms[i].Move);
            RootMoves.push_back(rm);
        }

        int multiPV = std::min(MultiPV, int(RootMoves.size()));
        RootMove lastBestRootMove = RootMove(Move::Null());

        int maxDepth = IsMain ? MaxDepth : MaxPly;
        while (++RootDepth < maxDepth)
        {
            //  The main thread is not allowed to search past info.MaxDepth
            if (IsMain && RootDepth > info.MaxDepth)
                break;

            if (StopSearching)
                break;

            for (RootMove& rm : RootMoves)
            {
                rm.PreviousScore = rm.Score;
            }

            for (PVIndex = 0; PVIndex < multiPV; PVIndex++)
            {
                if (StopSearching)
                    break;

                int alpha = AlphaStart;
                int beta = BetaStart;
                int window = ScoreInfinite;
                int score = RootMoves[PVIndex].PreviousScore;
                SelDepth = 0;

                if (RootDepth >= 5)
                {
                    window = AspirationWindowMargin;
                    alpha = std::max(AlphaStart, score - window);
                    beta = std::min(BetaStart, score + window);
                }

                while (true)
                {
                    score = Negamax<RootNode>(pos, ss, alpha, beta, std::max(1, RootDepth), false);

                    StableSort(RootMoves, PVIndex);

                    if (StopSearching)
                        break;

                    if (score <= alpha)
                    {
                        beta = (alpha + beta) / 2;
                        alpha = std::max(alpha - window, AlphaStart);
                    }
                    else if (score >= beta)
                    {
                        beta = std::min(beta + window, BetaStart);
                    }
                    else
                        break;

                    window += window / 2;
                }

                StableSort(RootMoves, 0);

                //  if (IsMain && (SearchPool.StopThreads || PVIndex == multiPV - 1 || tm.GetSearchTime() > 3000))
                if (IsMain)
                {
                    //info.OnDepthFinish?.Invoke(ref info);
                    //std::cout << "depth done" << std::endl;

                    RootMove rm = RootMoves[0];

                    if (StopSearching) {
                        rm = lastBestRootMove;
                    }

                    bool moveSearched = rm.Score != -ScoreInfinite;
                    int depth = moveSearched ? RootDepth : std::max(1, RootDepth - 1);
                    int moveScore = moveSearched ? rm.Score : rm.PreviousScore;

                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - TimeStart);
                    auto durCnt = std::max(1.0, (double) duration.count());
                    int nodesPerSec = (int)(Nodes / (durCnt / 1000));

                    std::cout << "info depth " << depth;
                    std::cout << " seldepth " << rm.Depth;
                    std::cout << " time " << durCnt;
                    std::cout << " score " << FormatMoveScore(moveScore);
                    std::cout << " nodes " << Nodes;
                    std::cout << " nps " << nodesPerSec;
                    std::cout << " pv";

                    for (int j = 0; j < rm.PV.size(); j++)
                    {
                        if (rm.PV[j] == Move::Null())
                        {
                            break;
                        }

                        std::cout << " " + rm.PV[j].SmithNotation(pos.IsChess960);
                    }


                    std::cout << std::endl;
                }
            }

            if (!IsMain)
                continue;

            if (StopSearching)
            {
                //  If we received a stop command or hit the hard time limit, our RootMoves may not have been filled in properly.
                //  In that case, we replace the current bestmove with the last depth's bestmove
                //  so that the move we send is based on an entire depth being searched instead of only a portion of it.
                RootMoves[0] = lastBestRootMove;

                for (int i = -10; i < MaxSearchStackPly; i++)
                {
                    AlignedFree((ss + i)->PV);
                }

                return;
            }

            lastBestRootMove.Move = RootMoves[0].Move;
            lastBestRootMove.Score = RootMoves[0].Score;
            lastBestRootMove.Depth = RootMoves[0].Depth;
            lastBestRootMove.PV.clear();
            for (int i = 0; i < RootMoves[0].PV.size(); i++)
            {
                lastBestRootMove.PV.push_back(RootMoves[0].PV[i]);
                if (lastBestRootMove.PV[i] == Move::Null())
                {
                    break;
                }
            }

            if (SoftTimeUp())
            {
                break;
            }

            if (!StopSearching)
            {
                CompletedDepth = RootDepth;
            }
        }

        if (IsMain && RootDepth >= MaxDepth && Nodes != MaxNodes && !StopSearching)
        {
            StopSearching = true;
        }

        for (int i = -10; i < MaxSearchStackPly; i++)
        {
            AlignedFree((ss + i)->PV);
        }
    }


    template <SearchNodeType NodeType>
    int SearchThread::Negamax(Position& pos, SearchStackEntry* ss, int alpha, int beta, int depth, bool cutNode) {
        constexpr bool isRoot = NodeType == SearchNodeType::RootNode;
        constexpr bool isPV = NodeType != SearchNodeType::NonPVNode;

        //std::cout << "Negamax(" << ss->Ply << ", " << alpha << ", " << beta << ", " << depth << ")" << std::endl;

        if (depth <= 0)
        {
            return QSearch<NodeType>(pos, ss, alpha, beta, depth);
        }

        Bitboard bb = pos.bb;
        //SearchThread thisThread = pos.Owner;
        HistoryTable* history = &History;

        ulong posHash = pos.State->Hash;

        Move bestMove = Move::Null();

        int ourColor = (int)pos.ToMove;
        int score = -ScoreMate - MaxPly;
        int bestScore = -ScoreInfinite;

        int startingAlpha = alpha;

        int probBeta = 0;

        short eval = ss->StaticEval;

        bool doSkip = ss->Skip != Move::Null();
        bool improving = false;
        TTEntry* tte = {};


        if (IsMain && ((++CheckupCount) >= CheckupMax))
        {
            CheckupCount = 0;

            //if (thisThread.CheckTime() || SearchPool.GetNodeCount() >= info.MaxNodes)
            //{
            //  SearchPool.StopThreads = true;
            //}

            if (CheckTime() || (Nodes >= MaxNodes))
            {
                StopSearching = true;
            }
        }

        if (isPV)
        {
            SelDepth = std::max(SelDepth, ss->Ply + 1);
        }

        if (!isRoot)
        {
            if (pos.IsDraw())
            {
                return ScoreDraw;
            }

            if (StopSearching || ss->Ply >= MaxSearchStackPly - 1)
            {
                if (pos.Checked())
                {
                    return ScoreDraw;
                }
                else
                {
                    return Horsie::NNUE::GetEvaluation(pos);
                }

            }

            //alpha = std::max(MakeMateScore(ss->Ply), alpha);
            //beta = std::min(ScoreMate - (ss->Ply + 1), beta);
            //if (alpha >= beta)
            //{
            //    return alpha;
            //}
        }

        (ss + 1)->Skip = Move::Null();
        (ss + 1)->Killer0 = (ss + 1)->Killer1 = Move::Null();
        ss->DoubleExtensions = (ss - 1)->DoubleExtensions;
        ss->StatScore = 0;
        ss->InCheck = pos.Checked();
        tte = TT.Probe(posHash, ss->TTHit);
        if (!doSkip)
        {
            ss->TTPV = isPV || (ss->TTHit && tte->PV());
        }

        short ttScore = ss->TTHit ? MakeNormalScore(tte->Score(), ss->Ply) : ScoreNone;

        Move ttMove = isRoot ? CurrentMove() : (ss->TTHit ? tte->BestMove : Move::Null());

        if (!isPV
            && !doSkip
            && tte->Depth() >= depth
            && ttScore != ScoreNone
            && (ttScore < alpha || cutNode)
            && (tte->Bound() & (ttScore >= beta ? BoundLower : BoundUpper)) != 0)
        {
            return ttScore;
        }

        if (ss->InCheck)
        {
            ss->StaticEval = eval = ScoreNone;
            goto MovesLoop;
        }

        if (doSkip)
        {
            eval = ss->StaticEval;
        }
        else if (ss->TTHit)
        {
            eval = ss->StaticEval = tte->StatEval() != ScoreNone ? tte->StatEval() : NNUE::GetEvaluation(pos);

            if (ttScore != ScoreNone && (tte->Bound() & (ttScore > eval ? BoundLower : BoundUpper)) != 0)
            {
                eval = ttScore;
            }
        }
        else
        {
            eval = ss->StaticEval = NNUE::GetEvaluation(pos);
        }

        if (ss->Ply >= 2)
        {
            improving = (ss - 2)->StaticEval != ScoreNone ? ss->StaticEval > (ss - 2)->StaticEval :
                       ((ss - 4)->StaticEval != ScoreNone ? ss->StaticEval > (ss - 4)->StaticEval : true);
        }


        if (UseReverseFutilityPruning
            && !ss->TTPV
            && !doSkip
            && depth <= ReverseFutilityPruningMaxDepth
            && ttMove == Move::Null()
            && (eval < ScoreAssuredWin)
            && (eval >= beta)
            && (eval - GetReverseFutilityMargin(depth, improving)) >= beta)
        {
            return (eval + beta) / 2;
        }

        if (UseNullMovePruning
            && !isPV
            && depth >= NMPMinDepth
            && eval >= beta
            && eval >= ss->StaticEval
            && !doSkip
            && (ss - 1)->CurrentMove != Move::Null()
            && pos.MaterialCountNonPawn[pos.ToMove] > 0)
        {
            int reduction = NMPReductionBase + (depth / NMPReductionDivisor);
            ss->CurrentMove = Move::Null();
            ss->ContinuationHistory = &history->Continuations[0][0].Histories[0][0];

            pos.MakeNullMove();
            score = -Negamax<NonPVNode>(pos, ss + 1, -beta, -beta + 1, depth - reduction, !cutNode);
            pos.UnmakeNullMove();

            if (score >= beta)
            {
                return score < ScoreTTWin ? score : beta;
            }
        }



        probBeta = beta + (improving ? ProbCutBetaImproving : ProbCutBeta);
        if (UseProbCut
            && !isPV
            && !doSkip
            && depth >= ProbCutMinDepth
            && std::abs(beta) < ScoreTTWin
            && (!ss->TTHit || tte->Depth() < depth - 3 || tte->Score() >= probBeta))
        {
            ScoredMove captures[MoveListSize] = {};
            int numCaps = Generate<GenLoud>(pos, captures, 0);
            AssignProbCutScores(pos, captures, numCaps);

            for (int i = 0; i < numCaps; i++)
            {
                Move m = OrderNextMove(captures, numCaps, i);
                if (!pos.IsLegal(m) || !pos.SEE_GE(m, 1))
                {
                    //  Skip illegal moves, and captures/promotions that don't result in a positive material trade
                    continue;
                }

                //int histIdx = PieceToHistory.GetIndex(ourColor, bb.GetPieceAtIndex(m.From()), m.To());
                int histIdx = MakePiece(ourColor, bb.GetPieceAtIndex(m.From()));
                bool isCap = (bb.GetPieceAtIndex(m.To()) != Piece::NONE && !m.IsCastle());

                prefetch(TT.GetCluster(pos.HashAfter(m)));
                ss->CurrentMove = m;
                //ss->ContinuationHistory = history.Continuations[ss->InCheck ? 1 : 0][(bb.GetPieceAtIndex(m.To()) != None && !m.IsCastle()) ? 1 : 0][histIdx];
                ss->ContinuationHistory = &history->Continuations[ss->InCheck ? 1 : 0][isCap ? 1 : 0].Histories[histIdx][m.To()];
                Nodes++;

                pos.MakeMove(m);

                score = -QSearch<NonPVNode>(pos, ss + 1, -probBeta, -probBeta + 1, DepthQChecks);

                if (score >= probBeta)
                {
                    //  Verify at a low depth
                    score = -Negamax<NonPVNode>(pos, ss + 1, -probBeta, -probBeta + 1, depth - 4, !cutNode);
                }

                pos.UnmakeMove(m);

                if (score >= probBeta)
                {
                    return score;
                }
            }
        }



        if (ttMove == Move::Null()
            && cutNode
            && depth >= ExtraCutNodeReductionMinDepth)
        {
            depth--;
        }

    MovesLoop:

        int legalMoves = 0, playedMoves = 0;
        int quietCount = 0, captureCount = 0;

        bool didSkip = false;

        Move captureMoves[16];
        Move quietMoves[16];

        bool skipQuiets = false;

        ScoredMove list[MoveListSize];
        int size = Generate<MoveGenType::GenNonEvasions>(pos, list, 0);
        AssignScores(pos, ss, *history, list, size, ttMove);

        for (int i = 0; i < size; i++)
        {
            Move m = OrderNextMove(list, size, i);

            if (m == ss->Skip)
            {
                didSkip = true;
                continue;
            }

            if (!pos.IsLegal(m))
            {
                continue;
            }

            int toSquare = m.To();
            bool isCapture = (bb.GetPieceAtIndex(toSquare) != Piece::NONE && !m.IsCastle());
            int thisPieceType = bb.GetPieceAtIndex(m.From());

            legalMoves++;
            int extend = 0;

            if (!isRoot
                && bestScore > ScoreMatedMax
                && pos.MaterialCountNonPawn[pos.ToMove] > 0)
            {
                if (skipQuiets == false)
                {
                    skipQuiets = legalMoves >= LMPTable[improving ? 1 : 0][depth];
                }

                bool givesCheck = ((pos.State->CheckSquares[thisPieceType] & SquareBB(toSquare)) != 0);

                if (skipQuiets && depth <= 8 && !(givesCheck || isCapture))
                {
                    continue;
                }

                if (givesCheck || isCapture || skipQuiets)
                {
                    if (!pos.SEE_GE(m, -LMRExchangeBase * depth))
                    {
                        continue;
                    }
                }
            }


            if (UseSingularExtensions
                && !isRoot
                && !doSkip
                && ss->Ply < RootDepth * 2
                && depth >= (SingularExtensionsMinDepth + (isPV && tte->PV() ? 1 : 0))
                && m == ttMove
                && std::abs(ttScore) < ScoreWin
                && ((tte->Bound() & BoundLower) != 0)
                && tte->Depth() >= depth - 3)
            {
                int singleBeta = ttScore - (SingularExtensionsNumerator * depth / 10);
                int singleDepth = (depth + SingularExtensionsDepthAugment) / 2;

                ss->Skip = m;
                score = Negamax<NonPVNode>(pos, ss, singleBeta - 1, singleBeta, singleDepth, cutNode);
                ss->Skip = Move::Null();

                if (score < singleBeta)
                {
                    extend = 1;

                    if (!isPV
                        && score < singleBeta - SingularExtensionsBeta
                        && ss->DoubleExtensions <= 8)
                    {
                        extend = 2;
                    }
                }
                else if (singleBeta >= beta)
                {
                    return singleBeta;
                }
                else if (ttScore >= beta || ttScore <= alpha)
                {
                    extend = isPV ? -1 : -extend;
                }
            }

            //int histIdx = PieceToHistory.GetIndex(ourColor, thisPieceType, toSquare);
            int histIdx = MakePiece(ourColor, thisPieceType);

            ss->DoubleExtensions = (ss - 1)->DoubleExtensions + (extend == 2 ? 1 : 0);

            prefetch(TT.GetCluster(pos.HashAfter(m)));
            ss->CurrentMove = m;
            //ss->ContinuationHistory = history.Continuations[ss->InCheck ? 1 : 0][isCapture ? 1 : 0][histIdx];
            ss->ContinuationHistory = &history->Continuations[ss->InCheck ? 1 : 0][isCapture ? 1 : 0].Histories[histIdx][toSquare];
            Nodes++;

            pos.MakeMove(m);
            
#if defined(_DEBUG) && defined(TREE)
            N_TABS(ss->Ply);
            std::cout << "NM " << m << std::endl;
#endif

            playedMoves++;
            ulong prevNodes = Nodes;

            if (isPV)
            {
                //System.Runtime.InteropServices.NativeMemory.Clear((ss + 1)->PV, (nuint)(MaxPly * sizeof(Move)));
                //std::memset((ss + 1)->PV, 0, (MaxPly * sizeof(Move)));
                (ss + 1)->PVLength = 0;
            }

            int newDepth = depth + extend;

            if (depth >= 2
                && legalMoves >= 2
                && !isCapture)
            {

                int R = LogarithmicReductionTable[depth][legalMoves];

                if (!improving)
                    R++;

                if (cutNode)
                    R++;

                if (isPV)
                    R--;

                if (m == ss->Killer0 || m == ss->Killer1)
                    R--;

                ss->StatScore = 2 * history->MainHistory[ourColor][m.GetMoveMask()] +
                                (*(ss - 1)->ContinuationHistory).history[histIdx][toSquare] +
                                (*(ss - 2)->ContinuationHistory).history[histIdx][toSquare] +
                                (*(ss - 4)->ContinuationHistory).history[histIdx][toSquare];

                R -= (ss->StatScore / (4096 * HistoryReductionMultiplier));

                //R = std::clamp(R, 1, newDepth);
                R = std::max(1, std::min(R, newDepth));
                int reducedDepth = (newDepth - R);

                score = -Negamax<NonPVNode>(pos, ss + 1, -alpha - 1, -alpha, reducedDepth, true);

                if (score > alpha && R > 1)
                {
                    newDepth += (score > (bestScore + LMRExtensionThreshold)) ? 1 : 0;
                    newDepth -= (score < (bestScore + newDepth)) ? 1 : 0;

                    if (newDepth - 1 > reducedDepth)
                    {
                        score = -Negamax<NonPVNode>(pos, ss + 1, -alpha - 1, -alpha, newDepth, !cutNode);
                    }

                    int bonus = 0;
                    if (score <= alpha)
                    {
                        bonus = -StatBonus(newDepth - 1);
                    }
                    else if (score >= beta)
                    {
                        bonus = StatBonus(newDepth - 1);
                    }

                    UpdateContinuations(ss, ourColor, thisPieceType, m.To(), bonus);
                }
            }
            else if (!isPV || legalMoves > 1)
            {
                score = -Negamax<NonPVNode>(pos, ss + 1, -alpha - 1, -alpha, newDepth - 1, !cutNode);
            }

            if (isPV && (playedMoves == 1 || score > alpha))
            {
                (ss + 1)->PV[0] = Move::Null();
                (ss + 1)->PVLength = 0;
                score = -Negamax<PVNode>(pos, ss + 1, -beta, -alpha, newDepth - 1, false);
            }

            pos.UnmakeMove(m);

            if (isRoot)
            {
                NodeTable[m.From()][m.To()] += Nodes - prevNodes;
            }


            if (StopSearching)
            {
                return ScoreDraw;
            }

            if (isRoot)
            {
                int rmIndex = 0;
                for (int j = 0; j < RootMoves.size(); j++)
                {
                    if (RootMoves[j].Move == m)
                    {
                        rmIndex = j;
                        break;
                    }
                }

                RootMove& rm = RootMoves[rmIndex];
                rm.AverageScore = (rm.AverageScore == -ScoreInfinite) ? score : ((rm.AverageScore + (score * 2)) / 3);

                if (playedMoves == 1 || score > alpha)
                {
                    rm.Score = score;
                    rm.Depth = SelDepth;

                    rm.PV.resize(1);
                    //Array.Fill(rm.PV, Move.Null, 1, MaxPly - rm.PVLength);

#if defined(NO_PV_LEN)
                    for (Move* childMove = (ss + 1)->PV; *childMove != Move::Null(); ++childMove)
                    {
                        //rm.PV[rm.PVLength++] = *childMove;
                        rm.PV.push_back(*childMove);
                    }
#else
                    for (int i = 0; i < (ss + 1)->PVLength; i++)
                    {
                        Move* childMove = ((ss + 1)->PV + i);

                        if (*childMove == Move::Null())
                            break;

                        rm.PV.push_back(*childMove);
                    }
#endif
                }
                else
                {
                    rm.Score = -ScoreInfinite;
                }
            }

            if (score > bestScore)
            {
                bestScore = score;

                if (score > alpha)
                {
                    bestMove = m;

                    if (isPV && !isRoot)
                    {
#if defined(NO_PV_LEN)
                        UpdatePV(ss->PV, m, (ss + 1)->PV);
#else
                        Move* pv = ss->PV;
                        Move* childPV = (ss + 1)->PV;

                        *pv++ = m;
                        ss->PVLength++;

                        if (childPV != nullptr) {
                            for (size_t j = 0; j < (ss + 1)->PVLength; j++)
                            {
                                if (*childPV == Move::Null())
                                    break;

                                *pv++ = *childPV++;
                                ss->PVLength++;
                            }
                        }
#endif
                    }

                    if (score >= beta)
                    {
                        UpdateStats(pos, ss, bestMove, bestScore, beta, depth, quietMoves, quietCount, captureMoves, captureCount);
                        break;
                    }

                    alpha = score;
                }
            }

            if (m != bestMove)
            {
                if (isCapture && captureCount < 16)
                {
                    captureMoves[captureCount++] = m;
                }
                else if (!isCapture && quietCount < 16)
                {
                    quietMoves[quietCount++] = m;
                }
            }
        }

        if (legalMoves == 0)
        {
            bestScore = ss->InCheck ? MakeMateScore(ss->Ply) : ScoreDraw;

            if (didSkip)
            {
                bestScore = alpha;
            }
        }

        if (bestScore <= alpha)
        {
            ss->TTPV = ss->TTPV || ((ss - 1)->TTPV && depth > 3);
        }

        if (!doSkip && !(isRoot && PVIndex > 0))
        {
            TTNodeType bound = (bestScore >= beta) ? TTNodeType::Alpha :
                      ((bestScore > startingAlpha) ? TTNodeType::Exact :
                                                     TTNodeType::Beta);

            Move moveToSave = (bound == TTNodeType::Beta) ? Move::Null() : bestMove;
            tte->Update(posHash, MakeTTScore((short)bestScore, ss->Ply), bound, depth, moveToSave, ss->StaticEval, ss->TTPV);
        }

        return bestScore;
    }


    template <SearchNodeType NodeType>
    int SearchThread::QSearch(Position& pos, SearchStackEntry* ss, int alpha, int beta, int depth) {
        
        constexpr bool isPV = NodeType != SearchNodeType::NonPVNode;

        //std::cout << "QSearch(" << ss->Ply << ", " << alpha << ", " << beta << ", " << depth << ")" << std::endl;

        Bitboard bb = pos.bb;
        SearchThread* thisThread = this;
        //HistoryTable history = this->History;
        Move bestMove = Move::Null();

        int score = -ScoreMate - MaxPly;
        short bestScore = -ScoreInfinite;
        short futility = -ScoreInfinite;

        short eval = ss->StaticEval;

        int startingAlpha = alpha;

        TTEntry* tte;

        ss->InCheck = pos.Checked();
        tte = TT.Probe(pos.State->Hash, ss->TTHit);
        int ttDepth = ss->InCheck || depth >= DepthQChecks ? DepthQChecks : DepthQNoChecks;
        short ttScore = ss->TTHit ? MakeNormalScore(tte->Score(), ss->Ply) : ScoreNone;
        Move ttMove = ss->TTHit ? tte->BestMove : Move::Null();
        bool ttPV = ss->TTHit && tte->PV();

        if (isPV)
        {
            ss->PV[0] = Move::Null();
        }


        if (pos.IsDraw())
        {
            return ScoreDraw;
        }

        if (ss->Ply >= MaxSearchStackPly - 1)
        {
            return ss->InCheck ? ScoreDraw : NNUE::GetEvaluation(pos);
        }

        if (isPV)
        {
            SelDepth = std::max(SelDepth, ss->Ply + 1);
        }

        if (!isPV
            && tte->Depth() >= ttDepth
            && ttScore != ScoreNone
            && (tte->Bound() & (ttScore >= beta ? BoundLower : BoundUpper)) != 0)
        {
            return ttScore;
        }

        if (ss->InCheck)
        {
            eval = ss->StaticEval = -ScoreInfinite;
        }
        else
        {
            if (ss->TTHit)
            {
                eval = ss->StaticEval = tte->StatEval();
                if (eval == ScoreNone)
                {
                    eval = ss->StaticEval = NNUE::GetEvaluation(pos);
                }

                if (ttScore != ScoreNone && ((tte->Bound() & (ttScore > eval ? BoundLower : BoundUpper)) != 0))
                {
                    eval = ttScore;
                }
            }
            else
            {
                if ((ss - 1)->CurrentMove == Move::Null())
                {
                    eval = ss->StaticEval = (short)(-(ss - 1)->StaticEval);
                }
                else
                {
                    eval = ss->StaticEval = NNUE::GetEvaluation(pos);
                }
            }

            if (eval >= beta)
            {
                return eval;
            }

            if (eval > alpha)
            {
                alpha = eval;
            }

            bestScore = eval;

            futility = (short)(std::min(ss->StaticEval, bestScore) + FutilityExchangeBase);
        }

        int prevSquare = (ss - 1)->CurrentMove == Move::Null() ? SQUARE_NB : (ss - 1)->CurrentMove.To();
        int legalMoves = 0;
        int movesMade = 0;
        int checkEvasions = 0;


        //int size = pos.GenPseudoLegalQS(list, ttDepth);
        //AssignQuiescenceScores(pos, ss, history, list, size, ttMove);

        ScoredMove list[MoveListSize] = {};
        int size = GenerateQS<MoveGenType::GenNonEvasions>(pos, list, ttDepth, 0);
        AssignQuiescenceScores(pos, ss, thisThread->History, list, size, ttMove);

        for (int i = 0; i < size; i++)
        {
            Move m = OrderNextMove(list, size, i);

            if (!pos.IsLegal(m))
            {
                continue;
            }

            legalMoves++;

            int moveFrom = m.From();
            int moveTo = m.To();
            int thisPiece = bb.GetPieceAtIndex(moveFrom);
            int capturedPiece = bb.GetPieceAtIndex(moveTo);

            bool isCapture = (capturedPiece != Piece::NONE && !m.IsCastle());
            bool isPromotion = m.IsPromotion();
            bool givesCheck = ((pos.State->CheckSquares[thisPiece] & SquareBB(moveTo)) != 0);

            movesMade++;

            if (bestScore > ScoreTTLoss)
            {
                if (!(givesCheck || isPromotion)
                    && (prevSquare != moveTo)
                    && futility > -ScoreWin)
                {
                    if (legalMoves > 3 && !ss->InCheck)
                    {
                        continue;
                    }

                    short futilityValue = (short)(futility + GetPieceValue(capturedPiece));

                    if (futilityValue <= alpha)
                    {
                        bestScore = std::max(bestScore, futilityValue);
                        continue;
                    }

                    if (futility <= alpha && !pos.SEE_GE(m, 1))
                    {
                        bestScore = std::max(bestScore, futility);
                        continue;
                    }
                }

                if (checkEvasions >= 2)
                {
                    break;
                }

                if (!ss->InCheck && !pos.SEE_GE(m, -90))
                {
                    continue;
                }
            }

            if (ss->InCheck && !isCapture)
            {
                checkEvasions++;
            }

            //int histIdx = PieceToHistory.GetIndex(pos.ToMove, pos.bb.GetPieceAtIndex(m.From), m.To);
            int histIdx = MakePiece(pos.ToMove, thisPiece);

            prefetch(TT.GetCluster(pos.HashAfter(m)));
            ss->CurrentMove = m;
            //ss->ContinuationHistory = history.Continuations[ss->InCheck ? 1 : 0][isCapture ? 1 : 0][histIdx];
            ss->ContinuationHistory = &thisThread->History.Continuations[ss->InCheck ? 1 : 0][isCapture ? 1 : 0].Histories[histIdx][moveTo];
            thisThread->Nodes++;

            pos.MakeMove(m);

#if defined(_DEBUG) && defined(TREE)
            N_TABS(ss->Ply);
            std::cout << "QS " << m << std::endl;
#endif

            score = -QSearch<NodeType>(pos, ss + 1, -beta, -alpha, depth - 1);
            pos.UnmakeMove(m);

            if (score > bestScore)
            {
                bestScore = (short)score;

                if (score > alpha)
                {
                    bestMove = m;
                    alpha = score;

                    if (isPV)
                    {
#if defined(NO_PV_LEN)
                        UpdatePV(ss->PV, m, (ss + 1)->PV);
#else
                        Move* pv = ss->PV;
                        Move* childPV = (ss + 1)->PV;

                        *pv++ = m;
                        ss->PVLength++;

                        if (childPV != nullptr) {
                            for (size_t j = 0; j < (ss + 1)->PVLength; j++)
                            {
                                if (*childPV == Move::Null())
                                    break;

                                *pv++ = *childPV++;
                                ss->PVLength++;
                            }
                        }
#endif
                    }

                    if (score >= beta)
                    {
                        break;
                    }
                }
            }
        }

        if (ss->InCheck && legalMoves == 0)
        {
            return MakeMateScore(ss->Ply);
        }

        TTNodeType bound = (bestScore >= beta) ? TTNodeType::Alpha :
                  ((bestScore > startingAlpha) ? TTNodeType::Exact : 
                                                 TTNodeType::Beta);

        tte->Update(pos.State->Hash, MakeTTScore(bestScore, ss->Ply), bound, depth, bestMove, ss->StaticEval, ss->TTPV);

        return bestScore;
    }






    void SearchThread::UpdateContinuations(SearchStackEntry* ss, int pc, int pt, int sq, int bonus) {
        for (int i : {1, 2, 4, 6}) {
            if (ss->InCheck && i > 2)
            {
                break;
            }

            if ((ss - i)->CurrentMove != Move::Null())
            {
                (*(ss - i)->ContinuationHistory).history[MakePiece(pc, pt)][sq] += (short)(bonus - 
               ((*(ss - i)->ContinuationHistory).history[MakePiece(pc, pt)][sq] * std::abs(bonus) / HistoryClamp));
            }
        }
    }


    void SearchThread::UpdateStats(Position& pos, SearchStackEntry* ss, Move bestMove, int bestScore, int beta, int depth, Move* quietMoves, int quietCount, Move* captureMoves, int captureCount) {
        
        int moveFrom = bestMove.From();
        int moveTo = bestMove.To();

        HistoryTable& history = this->History;
        Bitboard bb = pos.bb;

        int thisPiece = bb.GetPieceAtIndex(moveFrom);
        int thisColor = bb.GetColorAtIndex(moveFrom);
        int capturedPiece = bb.GetPieceAtIndex(moveTo);

        int quietMoveBonus = StatBonus(depth + 1);
        int quietMovePenalty = StatMalus(depth);

        if (capturedPiece != Piece::NONE && !bestMove.IsCastle())
        {
            //int idx = HistoryTable.CapIndex(thisColor, thisPiece, moveTo, capturedPiece);
            //history.ApplyBonus(history.CaptureHistory, idx, quietMoveBonus, HistoryTable.CaptureClamp);

            history.CaptureHistory[thisColor][thisPiece][moveTo][capturedPiece] += (short)(quietMoveBonus - 
           (history.CaptureHistory[thisColor][thisPiece][moveTo][capturedPiece] * std::abs(quietMoveBonus) / HistoryClamp));
            
        }
        else
        {

            int bestMoveBonus = (bestScore > beta + HistoryCaptureBonusMargin) ? quietMoveBonus : StatBonus(depth);

            if (ss->Killer0 != bestMove && !bestMove.IsEnPassant())
            {
                ss->Killer1 = ss->Killer0;
                ss->Killer0 = bestMove;
            }

            //history.ApplyBonus(history.MainHistory, HistoryTable.HistoryIndex(thisColor, bestMove), bestMoveBonus, HistoryTable.MainHistoryClamp);

            history.MainHistory[thisColor][bestMove.GetMoveMask()] += (short)(bestMoveBonus -
           (history.MainHistory[thisColor][bestMove.GetMoveMask()] * std::abs(bestMoveBonus) / HistoryClamp));

            UpdateContinuations(ss, thisColor, thisPiece, moveTo, bestMoveBonus);

            for (int i = 0; i < quietCount; i++)
            {
                Move m = quietMoves[i];
                //history.ApplyBonus(history.MainHistory, HistoryTable.HistoryIndex(thisColor, m), -quietMovePenalty, HistoryTable.MainHistoryClamp);

                history.MainHistory[thisColor][m.GetMoveMask()] += (short)(-quietMovePenalty -
               (history.MainHistory[thisColor][m.GetMoveMask()] * std::abs(-quietMovePenalty) / HistoryClamp));

                UpdateContinuations(ss, thisColor, bb.GetPieceAtIndex(m.From()), m.To(), -quietMovePenalty);
            }
        }

        for (int i = 0; i < captureCount; i++)
        {
            //int idx = HistoryTable.CapIndex(thisColor, bb.GetPieceAtIndex(captureMoves[i].From), captureMoves[i].To, bb.GetPieceAtIndex(captureMoves[i].To));
            //history.ApplyBonus(history.CaptureHistory, idx, -quietMoveBonus, HistoryTable.CaptureClamp);

            Move m = captureMoves[i];
            moveTo = m.To();

            int capThisPt = bb.GetPieceAtIndex(m.From());
            int capTheirPt = bb.GetPieceAtIndex(moveTo);

            history.CaptureHistory[thisColor][capThisPt][moveTo][capTheirPt] += (short)(-quietMoveBonus -
           (history.CaptureHistory[thisColor][capThisPt][moveTo][capTheirPt] * std::abs(-quietMoveBonus) / HistoryClamp));
        }

    }



    void SearchThread::UpdatePV(Move* pv, Move move, Move* childPV)
    {
        for (*pv++ = move; childPV != nullptr && *childPV != Move::Null();)
        {
            *pv++ = *childPV++;
        }
        *pv = Move::Null();
    }




    Move SearchThread::OrderNextMove(ScoredMove* moves, int size, int listIndex) {
        int max = INT32_MIN;
        int maxIndex = listIndex;

        for (int i = listIndex; i < size; i++)
        {
            if (moves[i].Score > max)
            {
                max = moves[i].Score;
                maxIndex = i;
            }
        }

        //(moves[maxIndex], moves[listIndex]) = (moves[listIndex], moves[maxIndex]);
        std::swap(moves[maxIndex], moves[listIndex]);

        return moves[listIndex].Move;
    }



    void SearchThread::AssignProbCutScores(Position& pos, ScoredMove* list, int size) {
        
        Bitboard& bb = pos.bb;
        for (int i = 0; i < size; i++)
        {
            Move m = list[i].Move;
            list[i].Score = GetSEEValue(m.IsEnPassant() ? PAWN : bb.GetPieceAtIndex(m.To()));
            if (m.IsPromotion())
            {
                //  Gives promotions a higher score than captures.
                //  We can assume a queen promotion is better than most captures.
                list[i].Score += GetSEEValue(QUEEN) + 1;
            }
        }
    }



    void SearchThread::AssignQuiescenceScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, int size, Move ttMove) {
        Bitboard bb = pos.bb;
        int pc = (int)pos.ToMove;

        for (int i = 0; i < size; i++) {
            Move m = list[i].Move;
            int moveTo = m.To();
            int moveFrom = m.From();

            if (m == ttMove)
            {
                list[i].Score = INT32_MAX - 100000;
            }
            else if (bb.GetPieceAtIndex(moveTo) != Piece::NONE && !m.IsCastle())
            {
                int capturedPiece = bb.GetPieceAtIndex(moveTo);
                auto hist = history.CaptureHistory[pc][bb.GetPieceAtIndex(moveFrom)][moveTo][capturedPiece];
                list[i].Score = (OrderingVictimValueMultiplier * GetPieceValue(capturedPiece)) + (hist / OrderingHistoryDivisor);
            }
            else {

                int pt = bb.GetPieceAtIndex(moveFrom);
                //int contIdx = PieceToHistory.GetIndex(pc, pt, moveTo);
                int contIdx = MakePiece(pc, pt);

                list[i].Score =  2 * history.MainHistory[pc][m.GetMoveMask()];
                list[i].Score += 2 * (*(ss - 1)->ContinuationHistory).history[contIdx][moveTo];
                list[i].Score +=     (*(ss - 2)->ContinuationHistory).history[contIdx][moveTo];
                list[i].Score +=     (*(ss - 4)->ContinuationHistory).history[contIdx][moveTo];
                list[i].Score +=     (*(ss - 6)->ContinuationHistory).history[contIdx][moveTo];

                if ((pos.State->CheckSquares[pt] & SquareBB(moveTo)) != 0)
                {
                    list[i].Score += OrderingGivesCheckBonus;
                }
            }
        }
    }



    void SearchThread::AssignScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, int size, Move ttMove) {
        
        Bitboard bb = pos.bb;
        int pc = (int) pos.ToMove;

        for (int i = 0; i < size; i++) {
            Move m = list[i].Move;
            int moveTo = m.To();
            int moveFrom = m.From();

            if (m == ttMove)
            {
                list[i].Score = INT32_MAX - 100000;
            }
            else if (m == ss->Killer0)
            {
                list[i].Score = INT32_MAX - 1000000;
            }
            else if (m == ss->Killer1)
            {
                list[i].Score = INT32_MAX - 2000000;
            }
            else if (bb.GetPieceAtIndex(moveTo) != Piece::NONE && !m.IsCastle())
            {
                int capturedPiece = bb.GetPieceAtIndex(moveTo);
                auto hist = history.CaptureHistory[pc][bb.GetPieceAtIndex(moveFrom)][moveTo][capturedPiece];
                list[i].Score = (OrderingVictimValueMultiplier * GetPieceValue(capturedPiece)) + (hist / OrderingHistoryDivisor);
            }
            else {

                int pt = bb.GetPieceAtIndex(moveFrom);
                //int contIdx = PieceToHistory.GetIndex(pc, pt, moveTo);
                int contIdx = MakePiece(pc, pt);

                list[i].Score =  2 * history.MainHistory[pc][m.GetMoveMask()];
                list[i].Score += 2 * (*(ss - 1)->ContinuationHistory).history[contIdx][moveTo];
                list[i].Score +=     (*(ss - 2)->ContinuationHistory).history[contIdx][moveTo];
                list[i].Score +=     (*(ss - 4)->ContinuationHistory).history[contIdx][moveTo];
                list[i].Score +=     (*(ss - 6)->ContinuationHistory).history[contIdx][moveTo];

                if ((pos.State->CheckSquares[pt] & SquareBB(moveTo)) != 0)
                {
                    list[i].Score += OrderingGivesCheckBonus;
                }
            }
        }
    }

    std::string SearchThread::Debug_GetMovesPlayed(SearchStackEntry* ss) {
        
        std::string moves = "";

        while (ss->Ply >= 0) {
            moves = ss->CurrentMove.SmithNotation(false) + " " + moves;
            ss--;
        }

        return moves;
    }

}