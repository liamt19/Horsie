
#define NO_PV_LEN 1
#define TREE
#undef TREE

#include "search.h"
#include "position.h"
#include "nnue/nn.h"
#include "tt.h"
#include "movegen.h"
#include "precomputed.h"

#include <chrono>
#include <iostream>

#include "search_options.h"


#include <string>
#include <fstream>

using namespace Horsie::Search;

static void appendLineToFile(std::string& filepath, std::string& line)
{
    std::ofstream file;
    //can't enable exception now because of gcc bug that raises ios_base::failure with useless message
    //file.exceptions(file.exceptions() | std::ios::failbit);
    file.open(filepath, std::ios::out | std::ios::app);
    if (file.fail())
        throw std::ios_base::failure(std::strerror(errno));

    //make sure write fails with exception if something is wrong
    file.exceptions(file.exceptions() | std::ios::failbit | std::ifstream::badbit);

    file << line << std::endl;
}

namespace Horsie {

    constexpr int MaxSearchStackPly = 256 - 10;
    constexpr int BoundLower = (int)TTNodeType::Alpha;
    constexpr int BoundUpper = (int)TTNodeType::Beta;

    constexpr double StabilityCoefficients[] = { 2.2, 1.6, 1.4, 1.1, 1, 0.95, 0.9 };
    constexpr int StabilityMax = 6;


    void SearchThread::Search(Position& pos, SearchLimits& info) {

        MaxNodes = info.MaxNodes;
        SearchTimeMS = info.MaxSearchTime;

        History.Clear();
        TT.TTUpdate();

        SearchStackEntry _SearchStackBlock[MaxPly] = {};
        SearchStackEntry* ss = &_SearchStackBlock[10];
        for (int i = -10; i < MaxSearchStackPly; i++)
        {
            (ss + i)->Clear();
            (ss + i)->Ply = (short)i;
            (ss + i)->PV = (Move*)AlignedAllocZeroed((MaxPly * sizeof(Move)), AllocAlignment);
            (ss + i)->PVLength = 0;
            (ss + i)->ContinuationHistory = &History.Continuations[0][0][0][0];
        }

        NNUE::ResetCaches(pos);

        NodeTable = {};

        RootMoves.clear();
        ScoredMove rms[MoveListSize] = {};
        int rmsSize = Generate<GenLegal>(pos, &rms[0], 0);
        for (size_t i = 0; i < rmsSize; i++)
        {
            RootMove rm = RootMove(rms[i].move);
            RootMoves.push_back(rm);
        }

        int multiPV = std::min(MultiPV, int(RootMoves.size()));

        std::vector<int> searchScores(MaxPly);

        RootMove lastBestRootMove = RootMove(Move::Null());
        int stability = 0;

        int maxDepth = IsMain ? MaxDepth : MaxPly;
        while (++RootDepth < maxDepth) {
            //  The main thread is not allowed to search past info.MaxDepth
            if (IsMain && RootDepth > info.MaxDepth)
                break;

            if (StopSearching)
                break;

            for (RootMove& rm : RootMoves) {
                rm.PreviousScore = rm.Score;
            }

            int usedDepth = RootDepth;

            for (PVIndex = 0; PVIndex < multiPV; PVIndex++) {
                if (StopSearching)
                    break;

                int alpha = AlphaStart;
                int beta = BetaStart;
                int window = ScoreInfinite;
                int score = RootMoves[PVIndex].AverageScore;
                SelDepth = 0;

                if (RootDepth >= 5) {
                    window = AspWindow;
                    alpha = std::max(AlphaStart, score - window);
                    beta = std::min(BetaStart, score + window);
                }

                while (true) {
                    score = Negamax<RootNode>(pos, ss, alpha, beta, std::max(1, usedDepth), false);

                    StableSort(RootMoves, PVIndex);

                    if (StopSearching)
                        break;

                    if (score <= alpha) {
                        beta = (alpha + beta) / 2;
                        alpha = std::max(alpha - window, AlphaStart);
                        usedDepth = RootDepth;
                    }
                    else if (score >= beta) {
                        beta = std::min(beta + window, BetaStart);
                        usedDepth = std::max(usedDepth - 1, RootDepth - 5);
                    }
                    else
                        break;

                    window += window / 2;
                }

                StableSort(RootMoves, 0);

                if (IsMain) {
                    if (OnDepthFinish) {
                        OnDepthFinish();
                    }
                    else {
                        RootMove& rm = RootMoves[0];

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
                                break;

                            std::cout << " " + rm.PV[j].SmithNotation(pos.IsChess960);
                        }

                        std::cout << std::endl;
                    }
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

            if (lastBestRootMove.move == RootMoves[0].move) {
                stability++;
            }
            else {
                stability = 0;
            }

            lastBestRootMove.move = RootMoves[0].move;
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

            searchScores.push_back(RootMoves[0].Score);

            if (HasSoftTime)
            {
                //  Base values taken from Clarity
                double multFactor = 1.0;
                if (RootDepth > 7) {
                    double nodeTM = (1.5 - NodeTable[RootMoves[0].move.From()][RootMoves[0].move.To()] / (double)Nodes) * 1.75;
                    double bmStability = StabilityCoefficients[std::min(stability, StabilityMax)];

                    double scoreStability = searchScores[searchScores.size() - 4]
                                          - searchScores[searchScores.size() - 1];

                    scoreStability = std::max(0.85, std::min(1.15, 0.034 * scoreStability));

                    multFactor = nodeTM * bmStability * scoreStability;
                }

                if (GetSearchTime() >= SoftTimeLimit * multFactor) {
                    break;
                }
            }

            if (Nodes >= info.SoftNodeLimit)
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

        if (depth <= 0) {
            return QSearch<NodeType>(pos, ss, alpha, beta);
        }

        if (!isRoot && alpha < ScoreDraw && pos.HasCycle(ss->Ply))
        {
            alpha = MakeDrawScore(Nodes);
            if (alpha >= beta)
                return alpha;
        }

        Bitboard& bb = pos.bb;
        HistoryTable* history = &History;

        Move bestMove = Move::Null();

        int us = pos.ToMove;
        int score = -ScoreMate - MaxPly;
        int bestScore = -ScoreInfinite;

        int startingAlpha = alpha;
        int probBeta;

        short rawEval = ScoreNone;
        short eval = ss->StaticEval;

        bool doSkip = ss->Skip != Move::Null();
        bool improving = false;
        TTEntry _tte{};
        TTEntry* tte = &_tte;


        if (IsMain && ((++CheckupCount) >= CheckupMax)) {
            CheckupCount = 0;

            //if (thisThread.CheckTime() || SearchPool.GetNodeCount() >= info.MaxNodes)
            //{
            //  SearchPool.StopThreads = true;
            //}

            if (CheckTime() || (Nodes >= MaxNodes)) {
                StopSearching = true;
            }
        }

        if (isPV) {
            SelDepth = std::max(SelDepth, ss->Ply + 1);
        }

        if (!isRoot) {
            if (pos.IsDraw())
            {
                return ScoreDraw;
            }

            if (StopSearching || ss->Ply >= MaxSearchStackPly - 1)
            {
                if (pos.Checked()) {
                    return ScoreDraw;
                }
                else {
                    return Horsie::NNUE::GetEvaluation(pos);
                }
            }

            alpha = std::max(MakeMateScore(ss->Ply), alpha);
            beta = std::min(ScoreMate - (ss->Ply + 1), beta);
            if (alpha >= beta) {
                return alpha;
            }
        }

        (ss + 1)->KillerMove = Move::Null();
        
        ss->DoubleExtensions = (ss - 1)->DoubleExtensions;
        ss->InCheck = pos.Checked();
        ss->TTHit = TT.Probe(pos.Hash(), tte);
        if (!doSkip) {
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

        if (ss->InCheck) {
            ss->StaticEval = eval = ScoreNone;
            goto MovesLoop;
        }

        if (doSkip) {
            eval = ss->StaticEval;
        }
        else if (ss->TTHit) {
            rawEval = tte->StatEval() != ScoreNone ? tte->StatEval() : NNUE::GetEvaluation(pos);

            eval = ss->StaticEval = AdjustEval(pos, us, rawEval);

            if (ttScore != ScoreNone && (tte->Bound() & (ttScore > eval ? BoundLower : BoundUpper)) != 0) {
                eval = ttScore;
            }
        }
        else {
            rawEval = NNUE::GetEvaluation(pos);

            eval = ss->StaticEval = AdjustEval(pos, us, rawEval);

            tte->Update(pos.Hash(), ScoreNone, TTNodeType::Invalid, TTEntry::DepthNone, Move::Null(), rawEval, ss->TTPV);
        }

        if (ss->Ply >= 2) {
            improving = (ss - 2)->StaticEval != ScoreNone ? ss->StaticEval > (ss - 2)->StaticEval :
                       ((ss - 4)->StaticEval != ScoreNone ? ss->StaticEval > (ss - 4)->StaticEval : true);
        }


        if (UseRFP
            && !ss->TTPV
            && !doSkip
            && depth <= RFPMaxDepth
            && ttMove == Move::Null()
            && (eval < ScoreAssuredWin)
            && (eval >= beta)
            && (eval - GetRFPMargin(depth, improving)) >= beta)
        {
            return (eval + beta) / 2;
        }

        if (UseNMP
            && !isPV
            && depth >= NMPMinDepth
            && eval >= beta
            && eval >= ss->StaticEval
            && !doSkip
            && (ss - 1)->CurrentMove != Move::Null()
            && pos.HasNonPawnMaterial(us)) 
        {
            int reduction = NMPBaseRed + (depth / NMPDepthDiv) + std::min((eval - beta) / NMPEvalDiv, NMPEvalMin);
            ss->CurrentMove = Move::Null();
            ss->ContinuationHistory = &history->Continuations[0][0][0][0];

            pos.MakeNullMove();
            prefetch(TT.GetCluster(pos.Hash()));
            score = -Negamax<NonPVNode>(pos, ss + 1, -beta, -beta + 1, depth - reduction, !cutNode);
            pos.UnmakeNullMove();

            if (score >= beta) {
                return score < ScoreTTWin ? score : beta;
            }
        }


        if (ttMove == Move::Null()
            && (cutNode || isPV)
            && depth >= IIRMinDepth)
        {
            depth--;
        }


        probBeta = beta + (improving ? ProbcutBetaImp : ProbCutBeta);
        if (UseProbCut
            && !isPV
            && !doSkip
            && depth >= ProbCutMinDepth
            && std::abs(beta) < ScoreTTWin
            && (!ss->TTHit || tte->Depth() < depth - 3 || tte->Score() >= probBeta))
        {
            ScoredMove captures[MoveListSize] = {};
            int numCaps = Generate<GenNoisy>(pos, captures, 0);
            AssignProbCutScores(pos, captures, numCaps);

            for (int i = 0; i < numCaps; i++) {
                Move m = OrderNextMove(captures, numCaps, i);
                if (!pos.IsLegal(m) || !pos.SEE_GE(m, std::max(1, probBeta - ss->StaticEval))) {
                    //  Skip illegal moves, and captures/promotions that don't result in a positive material trade
                    continue;
                }

                prefetch(TT.GetCluster(pos.HashAfter(m)));

                //int histIdx = PieceToHistory.GetIndex(ourColor, bb.GetPieceAtIndex(m.From()), m.To());
                int histIdx = MakePiece(us, bb.GetPieceAtIndex(m.From()));
                bool isCap = (bb.GetPieceAtIndex(m.To()) != Piece::NONE && !m.IsCastle());

                ss->CurrentMove = m;
                ss->ContinuationHistory = &history->Continuations[ss->InCheck][isCap][histIdx][m.To()];
                Nodes++;

                pos.MakeMove(m);

                score = -QSearch<NonPVNode>(pos, ss + 1, -probBeta, -probBeta + 1);

                if (score >= probBeta) {
                    //  Verify at a low depth
                    score = -Negamax<NonPVNode>(pos, ss + 1, -probBeta, -probBeta + 1, depth - 3, !cutNode);
                }

                pos.UnmakeMove(m);

                if (score >= probBeta) {
                    return score;
                }
            }
        }


        MovesLoop:

        //  "Small probcut idea" from SF: 
        //  https://github.com/official-stockfish/Stockfish/blob/7ccde25baf03e77926644b282fed68ba0b5ddf95/src/search.cpp#L878
        probBeta = beta + 435;
        if (ss->InCheck
            && !isPV
            && (ttMove != Move::Null() && bb.GetPieceAtIndex(ttMove.To()) != Piece::NONE)
            && ((tte->Bound() & BoundLower) != 0)
            && tte->Depth() >= depth - 6
            && ttScore >= probBeta
            && std::abs(ttScore) < ScoreTTWin
            && std::abs(beta) < ScoreTTWin)
        {
            return probBeta;
        }


        int legalMoves = 0, playedMoves = 0;
        int quietCount = 0, captureCount = 0;

        bool didSkip = false;
        int lmpMoves = LMPTable[improving][depth];

        Move captureMoves[16];
        Move quietMoves[16];

        bool skipQuiets = false;

        ScoredMove list[MoveListSize];
        int size = Generate<PseudoLegal>(pos, list, 0);
        AssignScores(pos, ss, *history, list, size, ttMove);

        for (int i = 0; i < size; i++) {
            Move m = OrderNextMove(list, size, i);

            if (m == ss->Skip) {
                didSkip = true;
                continue;
            }

            if (!pos.IsLegal(m)) {
                continue;
            }

            int moveFrom = m.From();
            int moveTo = m.To();
            int theirPiece = bb.GetPieceAtIndex(moveTo);
            int ourPiece = bb.GetPieceAtIndex(m.From());
            bool isCapture = (theirPiece != Piece::NONE && !m.IsCastle());
            
            legalMoves++;
            int extend = 0;


            if (ShallowPruning
                && !isRoot
                && bestScore > ScoreMatedMax
                && pos.HasNonPawnMaterial(us))
            {
                if (skipQuiets == false)
                {
                    skipQuiets = legalMoves >= lmpMoves;
                }

                bool givesCheck = ((pos.State->CheckSquares[ourPiece] & SquareBB(moveTo)) != 0);

                if (skipQuiets && depth <= ShallowMaxDepth && !(givesCheck || isCapture))
                {
                    continue;
                }

                if (givesCheck || isCapture || skipQuiets)
                {
                    if (!pos.SEE_GE(m, -ShallowSEEMargin * depth))
                    {
                        continue;
                    }
                }
            }


            if (UseSingularExtensions
                && !isRoot
                && !doSkip
                && ss->Ply < RootDepth * 2
                && depth >= (SEMinDepth + (isPV && tte->PV() ? 1 : 0))
                && m == ttMove
                && std::abs(ttScore) < ScoreWin
                && ((tte->Bound() & BoundLower) != 0)
                && tte->Depth() >= depth - 3)
            {
                int singleBeta = ttScore - (SENumerator * depth / 10);
                int singleDepth = (depth + SEDepthAdj) / 2;

                ss->Skip = m;
                score = Negamax<NonPVNode>(pos, ss, singleBeta - 1, singleBeta, singleDepth, cutNode);
                ss->Skip = Move::Null();

                if (score < singleBeta)
                {
                    bool doubleExt = !isPV && ss->DoubleExtensions <= 8 && (score < singleBeta - SEDoubleMargin);
                    bool tripleExt = doubleExt && (score < singleBeta - SETripleMargin - (isCapture * SETripleCapSub));

                    extend = 1 + doubleExt + tripleExt;
                }
                else if (singleBeta >= beta)
                {
                    return singleBeta;
                }
                else if (ttScore >= beta)
                {
                    extend = -2 + (isPV ? 1 : 0);
                }
                else if (cutNode)
                {
                    extend = -2;
                }
                else if (ttScore <= alpha)
                {
                    extend = -1;
                }
            }

            prefetch(TT.GetCluster(pos.HashAfter(m)));

            int histIdx = MakePiece(us, ourPiece);

            ss->DoubleExtensions = (short)((ss - 1)->DoubleExtensions + (extend >= 2 ? 1 : 0));
            ss->CurrentMove = m;
            ss->ContinuationHistory = &history->Continuations[ss->InCheck][isCapture][histIdx][moveTo];
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
                && !(isPV && isCapture))
            {

                int R = LogarithmicReductionTable[depth][legalMoves];

                R += (!improving);
                R += cutNode * 2;
                R -= ss->TTPV;
                R -= isPV;
                R -= (m == ss->KillerMove);

                int mHist = 2 * (isCapture ? history->CaptureHistory[us][ourPiece][moveTo][theirPiece] : history->MainHistory[us][m.GetMoveMask()]);    
                int histScore = mHist +
                                2 * (*(ss - 1)->ContinuationHistory)[histIdx][moveTo] +
                                    (*(ss - 2)->ContinuationHistory)[histIdx][moveTo] +
                                    (*(ss - 4)->ContinuationHistory)[histIdx][moveTo];

                R -= (histScore / (isCapture ? LMRCaptureDiv : LMRQuietDiv));

                R = std::max(1, std::min(R, newDepth));
                int reducedDepth = (newDepth - R);

                score = -Negamax<NonPVNode>(pos, ss + 1, -alpha - 1, -alpha, reducedDepth, true);

                if (score > alpha && R > 1) {
                    newDepth += (score > (bestScore + LMRExtMargin)) ? 1 : 0;
                    newDepth -= (score < (bestScore + newDepth)) ? 1 : 0;

                    if (newDepth - 1 > reducedDepth) {
                        score = -Negamax<NonPVNode>(pos, ss + 1, -alpha - 1, -alpha, newDepth - 1, !cutNode);
                    }

                    int bonus = 0;
                    if (score <= alpha) {
                        bonus = -StatBonus(newDepth - 1);
                    }
                    else if (score >= beta) {
                        bonus = StatBonus(newDepth - 1);
                    }

                    UpdateContinuations(ss, us, ourPiece, moveTo, bonus);
                }
            }
            else if (!isPV || legalMoves > 1) {
                score = -Negamax<NonPVNode>(pos, ss + 1, -alpha - 1, -alpha, newDepth - 1, !cutNode);
            }

            if (isPV && (playedMoves == 1 || score > alpha)) {
                (ss + 1)->PV[0] = Move::Null();
                (ss + 1)->PVLength = 0;
                score = -Negamax<PVNode>(pos, ss + 1, -beta, -alpha, newDepth - 1, false);
            }

            pos.UnmakeMove(m);

            if (isRoot) {
                NodeTable[moveFrom][moveTo] += Nodes - prevNodes;
            }

            if (StopSearching) {
                return ScoreDraw;
            }

            if (isRoot) {
                int rmIndex = 0;
                for (int j = 0; j < RootMoves.size(); j++) {
                    if (RootMoves[j].move == m) {
                        rmIndex = j;
                        break;
                    }
                }

                RootMove& rm = RootMoves[rmIndex];
                rm.AverageScore = (rm.AverageScore == -ScoreInfinite) ? score : ((rm.AverageScore + (score * 2)) / 3);

                if (playedMoves == 1 || score > alpha) {
                    rm.Score = score;
                    rm.Depth = SelDepth;

                    rm.PV.resize(1);

                    for (Move* childMove = (ss + 1)->PV; *childMove != Move::Null(); ++childMove) {
                        //rm.PV[rm.PVLength++] = *childMove;
                        rm.PV.push_back(*childMove);
                    }
                }
                else {
                    rm.Score = -ScoreInfinite;
                }
            }

            if (score > bestScore) {
                bestScore = score;

                if (score > alpha) {
                    bestMove = m;

                    if (isPV && !isRoot) {
                        UpdatePV(ss->PV, m, (ss + 1)->PV);
                    }

                    if (score >= beta) {
                        UpdateStats(pos, ss, bestMove, bestScore, beta, depth, quietMoves, quietCount, captureMoves, captureCount);
                        break;
                    }

                    alpha = score;
                }
            }

            if (m != bestMove) {
                if (isCapture && captureCount < 16) {
                    captureMoves[captureCount++] = m;
                }
                else if (!isCapture && quietCount < 16) {
                    quietMoves[quietCount++] = m;
                }
            }
        }

        if (legalMoves == 0) {
            bestScore = ss->InCheck ? MakeMateScore(ss->Ply) : ScoreDraw;

            if (didSkip) {
                bestScore = alpha;
            }
        }

        if (bestScore <= alpha) {
            ss->TTPV = ss->TTPV || ((ss - 1)->TTPV && depth > 3);
        }

        if (!doSkip && !(isRoot && PVIndex > 0)) {
            TTNodeType bound = (bestScore >= beta) ? TTNodeType::Alpha :
                      ((bestScore > startingAlpha) ? TTNodeType::Exact :
                                                     TTNodeType::Beta);

            Move moveToSave = (bound == TTNodeType::Beta) ? Move::Null() : bestMove;
            tte->Update(pos.Hash(), MakeTTScore((short)bestScore, ss->Ply), bound, depth, moveToSave, rawEval, ss->TTPV);

            if (!ss->InCheck
                && (bestMove.IsNull() || !pos.IsCapture(bestMove))
                && !(bound == TTNodeType::Alpha && bestScore <= ss->StaticEval)
                && !(bound == TTNodeType::Beta && bestScore >= ss->StaticEval))
            {
                auto diff = bestScore - ss->StaticEval;
                UpdateCorrectionHistory(pos, diff, depth);
            }
        }

        return bestScore;
    }


    template <SearchNodeType NodeType>
    int SearchThread::QSearch(Position& pos, SearchStackEntry* ss, int alpha, int beta) {
        
        constexpr bool isPV = NodeType != SearchNodeType::NonPVNode;

        if (alpha < ScoreDraw && pos.HasCycle(ss->Ply))
        {
            alpha = MakeDrawScore(Nodes);
            if (alpha >= beta)
                return alpha;
        }

        Bitboard& bb = pos.bb;
        SearchThread* thisThread = this;
        Move bestMove = Move::Null();

        const int us = pos.ToMove;
        int score = -ScoreMate - MaxPly;
        int bestScore = -ScoreInfinite;
        int futility = -ScoreInfinite;

        short rawEval = ScoreNone;
        short eval = ss->StaticEval;

        int startingAlpha = alpha;

        TTEntry _tte{};
        TTEntry* tte = &_tte;
        ss->InCheck = pos.Checked();
        ss->TTHit = TT.Probe(pos.Hash(), tte);
        short ttScore = ss->TTHit ? MakeNormalScore(tte->Score(), ss->Ply) : ScoreNone;
        Move ttMove = ss->TTHit ? tte->BestMove : Move::Null();
        bool ttPV = ss->TTHit && tte->PV();

        if (isPV) {
            ss->PV[0] = Move::Null();
            SelDepth = std::max(SelDepth, ss->Ply + 1);
        }


        if (pos.IsDraw()) {
            return ScoreDraw;
        }

        if (ss->Ply >= MaxSearchStackPly - 1) {
            return ss->InCheck ? ScoreDraw : NNUE::GetEvaluation(pos);
        }

        if (!isPV
            && ttScore != ScoreNone
            && (tte->Bound() & (ttScore >= beta ? BoundLower : BoundUpper)) != 0) {
            return ttScore;
        }

        if (ss->InCheck) {
            eval = ss->StaticEval = -ScoreInfinite;
        }
        else {
            if (ss->TTHit) {
                rawEval = (tte->StatEval() != ScoreNone) ? tte->StatEval() : NNUE::GetEvaluation(pos);

                eval = ss->StaticEval = AdjustEval(pos, us, rawEval);

                if (ttScore != ScoreNone && ((tte->Bound() & (ttScore > eval ? BoundLower : BoundUpper)) != 0)) {
                    eval = ttScore;
                }
            }
            else {
                rawEval = ((ss - 1)->CurrentMove == Move::Null()) ? (short)(-(ss - 1)->StaticEval) : NNUE::GetEvaluation(pos);

                eval = ss->StaticEval = AdjustEval(pos, us, rawEval);
            }

            if (eval >= beta) {
                if (!ss->TTHit)
                    tte->Update(pos.Hash(), MakeTTScore(eval, ss->Ply), TTNodeType::Alpha, TTEntry::DepthNone, Move::Null(), rawEval, false);

                if (std::abs(eval) < ScoreTTWin)
                    eval = (short)((4 * eval + beta) / 5);

                return eval;
            }

            if (eval > alpha) {
                alpha = eval;
            }

            bestScore = eval;

            futility = (std::min(ss->StaticEval, (short)bestScore) + QSFutileMargin);
        }

        int prevSquare = (ss - 1)->CurrentMove == Move::Null() ? SQUARE_NB : (ss - 1)->CurrentMove.To();
        int legalMoves = 0;
        int movesMade = 0;
        int checkEvasions = 0;

        ScoredMove list[MoveListSize] = {};
        int size = GenerateQS(pos, list, 0);
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
            int theirPiece = bb.GetPieceAtIndex(moveTo);
            int ourPiece = bb.GetPieceAtIndex(moveFrom);

            bool isCapture = (theirPiece != Piece::NONE && !m.IsCastle());
            bool givesCheck = ((pos.State->CheckSquares[ourPiece] & SquareBB(moveTo)) != 0);

            movesMade++;

            if (bestScore > ScoreTTLoss) {
                if (!(givesCheck || m.IsPromotion())
                    && (prevSquare != moveTo)
                    && futility > -ScoreWin) {

                    if (legalMoves > 3 && !ss->InCheck) {
                        continue;
                    }

                    int futilityValue = (futility + GetPieceValue(theirPiece));

                    if (futilityValue <= alpha) {
                        bestScore = std::max(bestScore, futilityValue);
                        continue;
                    }

                    if (futility <= alpha && !pos.SEE_GE(m, 1)) {
                        bestScore = std::max(bestScore, futility);
                        continue;
                    }
                }

                if (checkEvasions >= 2) {
                    break;
                }

                if (!ss->InCheck && !pos.SEE_GE(m, -QSSeeMargin)) {
                    continue;
                }
            }

            prefetch(TT.GetCluster(pos.HashAfter(m)));

            if (ss->InCheck && !isCapture) {
                checkEvasions++;
            }

            int histIdx = MakePiece(us, ourPiece);

            ss->CurrentMove = m;
            ss->ContinuationHistory = &thisThread->History.Continuations[ss->InCheck][isCapture][histIdx][moveTo];
            thisThread->Nodes++;

            pos.MakeMove(m);
            score = -QSearch<NodeType>(pos, ss + 1, -beta, -alpha);
            pos.UnmakeMove(m);

            if (score > bestScore) {
                bestScore = (short)score;

                if (score > alpha) {
                    bestMove = m;
                    alpha = score;

                    if (isPV) 
                        UpdatePV(ss->PV, m, (ss + 1)->PV);

                    if (score >= beta) {
                        if (std::abs(bestScore) < ScoreTTWin) 
                            bestScore = ((4 * bestScore + beta) / 5);

                        break;
                    }
                }
            }
        }

        if (ss->InCheck && legalMoves == 0)
            return MakeMateScore(ss->Ply);

        TTNodeType bound = (bestScore >= beta) ? TTNodeType::Alpha : TTNodeType::Beta;

        //auto rStr = std::to_string(pos.Hash()) + " " + std::to_string(bestScore);
        //auto fPath = std::string("temp.txt");
        //appendLineToFile(fPath, rStr);

        tte->Update(pos.Hash(), MakeTTScore((short)bestScore, ss->Ply), bound, 0, bestMove, rawEval, ss->TTPV);

        return bestScore;
    }




    void SearchThread::UpdatePV(Move* pv, Move move, Move* childPV) {
        for (*pv++ = move; childPV != nullptr && *childPV != Move::Null();) {
            *pv++ = *childPV++;
        }
        *pv = Move::Null();
    }


    void SearchThread::UpdateStats(Position& pos, SearchStackEntry* ss, Move bestMove, int bestScore, int beta, int depth, 
                                   Move* quietMoves, int quietCount, Move* captureMoves, int captureCount) {

        HistoryTable& history = this->History;
        int moveFrom = bestMove.From();
        int moveTo = bestMove.To();

        Bitboard& bb = pos.bb;

        int thisPiece = bb.GetPieceAtIndex(moveFrom);
        int thisColor = bb.GetColorAtIndex(moveFrom);
        int capturedPiece = bb.GetPieceAtIndex(moveTo);

        int bonus = StatBonus(depth);
        int malus = StatMalus(depth);

        if (capturedPiece != Piece::NONE && !bestMove.IsCastle())
        {
            //int idx = HistoryTable.CapIndex(thisColor, thisPiece, moveTo, capturedPiece);
            //history.ApplyBonus(history.CaptureHistory, idx, quietMoveBonus, HistoryTable.CaptureClamp);

           /*history.CaptureHistory[thisColor][thisPiece][moveTo][capturedPiece] += (short)(bonus -
           (history.CaptureHistory[thisColor][thisPiece][moveTo][capturedPiece] * std::abs(bonus) / HistoryClamp));*/
            history.CaptureHistory[thisColor][thisPiece][moveTo][capturedPiece] << bonus;
            
        }
        else
        {
            if (!bestMove.IsEnPassant())
            {
                ss->KillerMove = bestMove;
            }

            //  Idea from Ethereal:
            //  Don't reward/punish moves resulting from a trivial, low-depth cutoff
            if (quietCount == 0 && depth <= 3)
            {
                return;
            }

            /*history.MainHistory[thisColor][bestMove.GetMoveMask()] += (short)(bonus -
            (history.MainHistory[thisColor][bestMove.GetMoveMask()] * std::abs(bonus) / HistoryClamp));*/
            history.MainHistory[thisColor][bestMove.GetMoveMask()] << bonus;
            UpdateContinuations(ss, thisColor, thisPiece, moveTo, bonus);


            for (int i = 0; i < quietCount; i++) {
                Move m = quietMoves[i];
                thisPiece = bb.GetPieceAtIndex(m.From());

                /*history.MainHistory[thisColor][m.GetMoveMask()] += (short)(-malus -
               (history.MainHistory[thisColor][m.GetMoveMask()] * std::abs(-malus) / HistoryClamp));*/
                history.MainHistory[thisColor][m.GetMoveMask()] << -malus;

                UpdateContinuations(ss, thisColor, thisPiece, m.To(), -malus);
            }
        }

        for (int i = 0; i < captureCount; i++) {
            Move m = captureMoves[i];
            thisPiece = bb.GetPieceAtIndex(m.From());
            capturedPiece = bb.GetPieceAtIndex(m.To());

            /*history.CaptureHistory[thisColor][thisPiece][moveTo][capturedPiece] += (short)(-malus -
           (history.CaptureHistory[thisColor][thisPiece][moveTo][capturedPiece] * std::abs(-malus) / HistoryClamp));*/
            history.CaptureHistory[thisColor][thisPiece][m.To()][capturedPiece] << -malus;
        }

    }


    short SearchThread::AdjustEval(Position& pos, int us, short rawEval) {
        rawEval = (short)(rawEval * (200 - pos.State->HalfmoveClock) / 200);

        const auto pawn = History.PawnCorrection[us][pos.PawnHash() % 16384] / CorrectionGrain;
        const auto nonPawn = (History.NonPawnCorrection[us][pos.NonPawnHash(Color::WHITE) % 16384] + History.NonPawnCorrection[us][pos.NonPawnHash(Color::BLACK) % 16384]) / CorrectionGrain;
        const auto corr = (pawn * 200 + nonPawn * 100) / 300;

        return (short)(rawEval + corr);
    }


    void SearchThread::UpdateCorrectionHistory(Position& pos, int diff, int depth) {
        const auto scaledWeight = std::min((depth * depth) + 1, 128);

        auto& pawnCh = History.PawnCorrection[pos.ToMove][pos.PawnHash() % 16384];
        const auto pawnBonus = (pawnCh * (CorrectionScale - scaledWeight) + (diff * CorrectionGrain * scaledWeight)) / CorrectionScale;
        pawnCh = (short)std::clamp(pawnBonus, -CorrectionMax, CorrectionMax);

        auto& nonPawnChW = History.NonPawnCorrection[pos.ToMove][pos.NonPawnHash(Color::WHITE) % 16384];
        const auto nonPawnBonusW = (nonPawnChW * (CorrectionScale - scaledWeight) + (diff * CorrectionGrain * scaledWeight)) / CorrectionScale;
        nonPawnChW = std::clamp(nonPawnBonusW, -CorrectionMax, CorrectionMax);

        auto& nonPawnChB = History.NonPawnCorrection[pos.ToMove][pos.NonPawnHash(Color::BLACK) % 16384];
        const auto nonPawnBonusB = (nonPawnChB * (CorrectionScale - scaledWeight) + (diff * CorrectionGrain * scaledWeight)) / CorrectionScale;
        nonPawnChB = std::clamp(nonPawnBonusB, -CorrectionMax, CorrectionMax);
    }


    void SearchThread::UpdateContinuations(SearchStackEntry* ss, int pc, int pt, int sq, int bonus) {
        auto piece = MakePiece(pc, pt);

        for (int i : {1, 2, 4, 6}) {
            if (ss->InCheck && i > 2)
            {
                break;
            }

            if ((ss - i)->CurrentMove != Move::Null())
            {
                (*(ss - i)->ContinuationHistory)[piece][sq] << bonus;
            }
        }
    }





    Move SearchThread::OrderNextMove(ScoredMove* moves, int size, int listIndex) {
        int max = INT32_MIN;
        int maxIndex = listIndex;

        for (int i = listIndex; i < size; i++) {
            if (moves[i].Score > max) {
                max = moves[i].Score;
                maxIndex = i;
            }
        }

        //(moves[maxIndex], moves[listIndex]) = (moves[listIndex], moves[maxIndex]);
        std::swap(moves[maxIndex], moves[listIndex]);

        return moves[listIndex].move;
    }



    void SearchThread::AssignProbCutScores(Position& pos, ScoredMove* list, int size) {
        
        Bitboard& bb = pos.bb;
        for (int i = 0; i < size; i++) {
            Move m = list[i].move;

            list[i].Score = GetSEEValue(m.IsEnPassant() ? PAWN : bb.GetPieceAtIndex(m.To()));
            if (m.IsPromotion()) {
                //  Gives promotions a higher score than captures.
                //  We can assume a queen promotion is better than most captures.
                list[i].Score += GetSEEValue(QUEEN) + 1;
            }
        }
    }



    void SearchThread::AssignQuiescenceScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, int size, Move ttMove) {
        Bitboard& bb = pos.bb;
        int pc = (int)pos.ToMove;

        for (int i = 0; i < size; i++) {
            Move m = list[i].move;
            int moveTo = m.To();
            int moveFrom = m.From();
            int pt = bb.GetPieceAtIndex(moveFrom);

            if (m == ttMove) {
                list[i].Score = INT32_MAX - 100000;
            }
            else if (bb.GetPieceAtIndex(moveTo) != Piece::NONE && !m.IsCastle()) {
                int capturedPiece = bb.GetPieceAtIndex(moveTo);
                auto& hist = history.CaptureHistory[pc][pt][moveTo][capturedPiece];
                list[i].Score = (OrderingVictimValueMultiplier * GetPieceValue(capturedPiece)) + hist;
            }
            else {
                int contIdx = MakePiece(pc, pt);

                list[i].Score =  2 * history.MainHistory[pc][m.GetMoveMask()];
                list[i].Score += 2 * (*(ss - 1)->ContinuationHistory)[contIdx][moveTo];
                list[i].Score +=     (*(ss - 2)->ContinuationHistory)[contIdx][moveTo];
                list[i].Score +=     (*(ss - 4)->ContinuationHistory)[contIdx][moveTo];
                list[i].Score +=     (*(ss - 6)->ContinuationHistory)[contIdx][moveTo];

                if ((pos.State->CheckSquares[pt] & SquareBB(moveTo)) != 0) {
                    list[i].Score += OrderingGivesCheckBonus;
                }
            }

            if (pt == Knight) {
                list[i].Score += 200;
            }
        }
    }



    void SearchThread::AssignScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, int size, Move ttMove) {
        
        Bitboard& bb = pos.bb;
        int pc = (int) pos.ToMove;

        for (int i = 0; i < size; i++) {
            Move m = list[i].move;
            int moveTo = m.To();
            int moveFrom = m.From();
            int pt = bb.GetPieceAtIndex(moveFrom);

            if (m == ttMove) {
                list[i].Score = INT32_MAX - 100000;
            }
            else if (m == ss->KillerMove) {
                list[i].Score = INT32_MAX - 1000000;
            }
            else if (bb.GetPieceAtIndex(moveTo) != Piece::NONE && !m.IsCastle()) {
                int capturedPiece = bb.GetPieceAtIndex(moveTo);
                auto& hist = history.CaptureHistory[pc][pt][moveTo][capturedPiece];
                list[i].Score = (OrderingVictimValueMultiplier * GetPieceValue(capturedPiece)) + hist;
            }
            else {
                int contIdx = MakePiece(pc, pt);

                list[i].Score =  2 * history.MainHistory[pc][m.GetMoveMask()];
                list[i].Score += 2 * (*(ss - 1)->ContinuationHistory)[contIdx][moveTo];
                list[i].Score +=     (*(ss - 2)->ContinuationHistory)[contIdx][moveTo];
                list[i].Score +=     (*(ss - 4)->ContinuationHistory)[contIdx][moveTo];
                list[i].Score +=     (*(ss - 6)->ContinuationHistory)[contIdx][moveTo];

                if ((pos.State->CheckSquares[pt] & SquareBB(moveTo)) != 0) {
                    list[i].Score += OrderingGivesCheckBonus;
                }
            }

            if (pt == Knight) {
                list[i].Score += 200;
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