
#include "search.h"
#include "position.h"
#include "nn.h"
#include "tt.h"
#include "movegen.h"
#include "precomputed.h"

#include "search_options.h"

using namespace Horsie::Search;

namespace Horsie {

    constexpr int MaxSearchStackPly = 256 - 10;
    constexpr int DepthQChecks = 0;
    constexpr int DepthQNoChecks = -1;

    constexpr int BoundLower = (int)TTNodeType::Alpha;
    constexpr int BoundUpper = (int)TTNodeType::Beta;


    void SearchThread::Search(Position& pos) {

        SearchLimits info = SearchLimits();
        info.MaxDepth = 3;

        SearchStackEntry _SearchStackBlock[MaxPly] = {};
        SearchStackEntry* ss = &_SearchStackBlock[10];
        for (int i = -10; i < MaxSearchStackPly; i++)
        {
            (ss + i)->Clear();
            (ss + i)->Ply = (short)i;
            (ss + i)->PV = (Move*)AlignedAllocZeroed((MaxPly * sizeof(Move)), AllocAlignment);
            (ss + i)->ContinuationHistory = History.Continuations[0][0].Histories[0][0];
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

            for (RootMove rm : RootMoves)
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
                    std::cout << "depth done" << std::endl;
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

            for (int i = 0; i < MaxPly; i++)
            {
                lastBestRootMove.PV[i] = RootMoves[0].PV[i];
                if (lastBestRootMove.PV[i] == Move::Null())
                {
                    break;
                }
            }

            //if (SoftTimeUp(tm))
            //{
            //    break;
            //}

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

        std::cout << "Negamax(" << ss->Ply << ", " << alpha << ", " << beta << ", " << depth << ")" << std::endl;

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

        short eval = ss->StaticEval;

        bool doSkip = ss->Skip != Move::Null();
        bool improving = false;
        TTEntry* tte = {};


        if (IsMain && ((++CheckupCount) >= CheckupMax))
        {
            CheckupCount = 0;

            //if (thisThread.CheckTime() || SearchPool.GetNodeCount() >= info.MaxNodes)
            //{
            //	SearchPool.StopThreads = true;
            //}

            if (CheckTime())
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

            alpha = std::max(MakeMateScore(ss->Ply), alpha);
            beta = std::min(ScoreMate - (ss->Ply + 1), beta);
            if (alpha >= beta)
            {
                return alpha;
            }
        }

        (ss + 1)->Skip = Move::Null();
        (ss + 1)->Killer0 = (ss + 1)->Killer1 = Move::Null();
        ss->DoubleExtensions = (ss - 1)->DoubleExtensions;
        ss->StatScore = 0;
        ss->InCheck = pos.Checked();
        ss->TTHit = TT.Probe(posHash, tte);
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
            ss->ContinuationHistory = history->Continuations[0][0].Histories[0][0];

            pos.MakeNullMove();
            score = -Negamax<NonPVNode>(pos, ss + 1, -beta, -beta + 1, depth - reduction, !cutNode);
            pos.UnmakeNullMove();

            if (score >= beta)
            {
                return score < ScoreTTWin ? score : beta;
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

        Move captureMoves[16] = {};
        Move quietMoves[16] = {};

        bool skipQuiets = false;

        ScoredMove list[MoveListSize] = {};
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
                    if (!SEE_GE(pos, m, -LMRExchangeBase * depth))
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
            ss->ContinuationHistory = history->Continuations[ss->InCheck ? 1 : 0][isCapture ? 1 : 0].Histories[histIdx][toSquare];
            Nodes++;

            pos.MakeMove(m);

            playedMoves++;
            ulong prevNodes = Nodes;

            if (isPV)
            {
                //System.Runtime.InteropServices.NativeMemory.Clear((ss + 1)->PV, (nuint)(MaxPly * sizeof(Move)));
                std::memset((ss + 1)->PV, 0, (MaxPly * sizeof(Move)));
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

                RootMove rm = RootMoves[rmIndex];
                rm.AverageScore = (rm.AverageScore == -ScoreInfinite) ? score : ((rm.AverageScore + (score * 2)) / 3);

                if (playedMoves == 1 || score > alpha)
                {
                    rm.Score = score;
                    rm.Depth = SelDepth;

                    rm.PVLength = 1;
                    //Array.Fill(rm.PV, Move.Null, 1, MaxPly - rm.PVLength);
                    for (size_t i = 1; i < MaxPly - rm.PVLength; i++)
                    {
                        rm.PV[i] = Move::Null();
                    }
                    
                    for (Move* childMove = (ss + 1)->PV; *childMove != Move::Null(); ++childMove)
                    {
                        rm.PV[rm.PVLength++] = *childMove;
                    }
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
                        UpdatePV(ss->PV, m, (ss + 1)->PV);
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

        std::cout << "QSearch(" << ss->Ply << ", " << alpha << ", " << beta << ", " << depth << ")" << std::endl;

        Bitboard bb = pos.bb;
        //SearchThread thisThread = pos.Owner;
        SearchThread thisThread = *this;
        HistoryTable history = this->History;
        Move bestMove = Move::Null();

        int score = -ScoreMate - MaxPly;
        short bestScore = -ScoreInfinite;
        short futility = -ScoreInfinite;

        short eval = ss->StaticEval;

        int startingAlpha = alpha;

        TTEntry* tte = {};

        ss->InCheck = pos.Checked();
        ss->TTHit = TT.Probe(pos.State->Hash, tte);
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
            thisThread.SelDepth = std::max(thisThread.SelDepth, ss->Ply + 1);
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
        int size = Generate<MoveGenType::GenNonEvasions>(pos, list, 0);
        AssignScores(pos, ss, history, list, size, ttMove);

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

                    if (futility <= alpha && !SEE_GE(pos, m, 1))
                    {
                        bestScore = std::max(bestScore, futility);
                        continue;
                    }
                }

                if (checkEvasions >= 2)
                {
                    break;
                }

                if (!ss->InCheck && !SEE_GE(pos, m, -90))
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
            ss->ContinuationHistory = history.Continuations[ss->InCheck ? 1 : 0][isCapture ? 1 : 0].Histories[histIdx][moveTo];
            thisThread.Nodes++;

            pos.MakeMove(m);
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
                        UpdatePV(ss->PV, m, (ss + 1)->PV);
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


    void SearchThread::UpdateStats(Position pos, SearchStackEntry* ss, Move bestMove, int bestScore, int beta, int depth, Move* quietMoves, int quietCount, Move* captureMoves, int captureCount) {

        HistoryTable history = this->History;
        int moveFrom = bestMove.From();
        int moveTo = bestMove.To();

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
                int contIdx = (pc * PIECE_NB) + pt;

                list[i].Score =  2 * history.MainHistory[pc][m.GetMoveMask()];
                list[i].Score += 2 * (*(ss - 1)->ContinuationHistory).history[contIdx][moveTo];
                list[i].Score +=		(*(ss - 2)->ContinuationHistory).history[contIdx][moveTo];
                list[i].Score +=		(*(ss - 4)->ContinuationHistory).history[contIdx][moveTo];
                list[i].Score +=		(*(ss - 6)->ContinuationHistory).history[contIdx][moveTo];

                if ((pos.State->CheckSquares[pt] & SquareBB(moveTo)) != 0)
                {
                    list[i].Score += OrderingGivesCheckBonus;
                }
            }
        }
    }





    bool SearchThread::SEE_GE(Position pos, Move m, int threshold)
    {
        if (m.IsCastle() || m.IsEnPassant() || m.IsPromotion())
        {
            return threshold <= 0;
        }

        Bitboard bb = pos.bb;

        int from = m.From();
        int to = m.To();

        int swap = GetSEEValue(bb.PieceTypes[to]) - threshold;
        if (swap < 0)
            return false;

        swap = GetSEEValue(bb.PieceTypes[from]) - swap;
        if (swap <= 0)
            return true;

        ulong occ = (bb.Occupancy ^ SquareBB(from) | SquareBB(to));

        ulong attackers = bb.AttackersTo(to, occ);
        ulong stmAttackers;
        ulong temp;

        int stm = pos.ToMove;
        int res = 1;
        while (true)
        {
            stm = Not(stm);
            attackers &= occ;

            stmAttackers = attackers & bb.Colors[stm];
            if (stmAttackers == 0)
            {
                break;
            }

            if ((pos.State->Pinners[Not(stm)] & occ) != 0)
            {
                stmAttackers &= ~pos.State->BlockingPieces[stm];
                if (stmAttackers == 0)
                {
                    break;
                }
            }

            res ^= 1;

            if ((temp = stmAttackers & bb.Pieces[Pawn]) != 0)
            {
                occ ^= SquareBB(lsb(temp));
                if ((swap = GetPieceValue(Pawn) - swap) < res)
                    break;

                attackers |= attacks_bb<BISHOP>(to, occ) & (bb.Pieces[Bishop] | bb.Pieces[Queen]);
            }
            else if ((temp = stmAttackers & bb.Pieces[Knight]) != 0)
            {
                occ ^= SquareBB(lsb(temp));
                if ((swap = GetPieceValue(Knight) - swap) < res)
                    break;
            }
            else if ((temp = stmAttackers & bb.Pieces[Bishop]) != 0)
            {
                occ ^= SquareBB(lsb(temp));
                if ((swap = GetPieceValue(BISHOP) - swap) < res)
                    break;

                attackers |= attacks_bb<BISHOP>(to, occ) & (bb.Pieces[Bishop] | bb.Pieces[Queen]);
            }
            else if ((temp = stmAttackers & bb.Pieces[Rook]) != 0)
            {
                occ ^= SquareBB(lsb(temp));
                if ((swap = GetPieceValue(ROOK) - swap) < res)
                    break;

                attackers |= attacks_bb<ROOK>(to, occ) & (bb.Pieces[Rook] | bb.Pieces[Queen]);
            }
            else if ((temp = stmAttackers & bb.Pieces[Queen]) != 0)
            {
                occ ^= SquareBB(lsb(temp));
                if ((swap = GetPieceValue(Queen) - swap) < res)
                    break;

                attackers |= (attacks_bb<BISHOP>(to, occ) & (bb.Pieces[Bishop] | bb.Pieces[Queen])) | (attacks_bb<ROOK>(to, occ) & (bb.Pieces[Rook] | bb.Pieces[Queen]));
            }
            else
            {
                if ((attackers & ~bb.Pieces[stm]) != 0)
                {
                    return (res ^ 1) != 0;
                }
                else
                {
                    return res != 0;
                }
            }
        }

        return res != 0;
    }



}