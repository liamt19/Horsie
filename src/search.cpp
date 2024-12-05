
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

namespace Horsie {

    constexpr i32 MaxSearchStackPly = 256 - 10;
    constexpr i32 BoundLower = static_cast<i32>(TTNodeType::Alpha);
    constexpr i32 BoundUpper = static_cast<i32>(TTNodeType::Beta);

    constexpr double StabilityCoefficients[] = { 2.2, 1.6, 1.4, 1.1, 1, 0.95, 0.9 };
    constexpr i32 StabilityMax = 6;


    void SearchThread::Search(Position& pos, SearchLimits& info) {

        MaxNodes = info.MaxNodes;
        SearchTimeMS = info.MaxSearchTime;

        TT.TTUpdate();

        SearchStackEntry _SearchStackBlock[MaxPly] = {};
        SearchStackEntry* ss = &_SearchStackBlock[10];
        for (i32 i = -10; i < MaxSearchStackPly; i++)
        {
            (ss + i)->Clear();
            (ss + i)->Ply = static_cast<i16>(i);
            (ss + i)->PV = AlignedAlloc<Move>(MaxPly);
            (ss + i)->PVLength = 0;
            (ss + i)->ContinuationHistory = &History.Continuations[0][0][0][0];
        }

        NNUE::ResetCaches(pos);

        NodeTable = {};

        RootMoves.clear();
        ScoredMove rms[MoveListSize] = {};
        i32 rmsSize = Generate<GenLegal>(pos, &rms[0], 0);
        for (size_t i = 0; i < rmsSize; i++)
        {
            RootMove rm = RootMove(rms[i].move);
            RootMoves.push_back(rm);
        }

        i32 multiPV = std::min(MultiPV, i32(RootMoves.size()));

        std::vector<i32> searchScores(MaxPly);

        RootMove lastBestRootMove = RootMove(Move::Null());
        i32 stability = 0;

        i32 maxDepth = IsMain ? MaxDepth : MaxPly;
        while (++RootDepth < maxDepth) {
            //  The main thread is not allowed to search past info.MaxDepth
            if (IsMain && RootDepth > info.MaxDepth)
                break;

            if (StopSearching)
                break;

            for (RootMove& rm : RootMoves) {
                rm.PreviousScore = rm.Score;
            }

            i32 usedDepth = RootDepth;

            for (PVIndex = 0; PVIndex < multiPV; PVIndex++) {
                if (StopSearching)
                    break;

                i32 alpha = AlphaStart;
                i32 beta = BetaStart;
                i32 window = ScoreInfinite;
                i32 score = RootMoves[PVIndex].AverageScore;
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

                        bool moveSearched = rm.Score != -ScoreInfinite;
                        i32 depth = moveSearched ? RootDepth : std::max(1, RootDepth - 1);
                        i32 moveScore = moveSearched ? rm.Score : rm.PreviousScore;

                        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - TimeStart);
                        auto durCnt = std::max(1.0, (double) duration.count());
                        i32 nodesPerSec = static_cast<i32>(Nodes / (durCnt / 1000));

                        std::cout << "info depth " << depth;
                        std::cout << " seldepth " << rm.Depth;
                        std::cout << " time " << durCnt;
                        std::cout << " score " << FormatMoveScore(moveScore);
                        std::cout << " nodes " << Nodes;
                        std::cout << " nps " << nodesPerSec;
                        std::cout << " pv";

                        for (i32 j = 0; j < rm.PV.size(); j++)
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

                for (i32 i = -10; i < MaxSearchStackPly; i++)
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
            for (i32 i = 0; i < RootMoves[0].PV.size(); i++)
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
                    const auto [bmFrom, bmTo] = RootMoves[0].move.Unpack();

                    double nodeTM = (1.5 - NodeTable[bmFrom][bmTo] / (double)Nodes) * 1.75;
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

        for (i32 i = -10; i < MaxSearchStackPly; i++)
        {
            AlignedFree((ss + i)->PV);
        }
    }


    template <SearchNodeType NodeType>
    i32 SearchThread::Negamax(Position& pos, SearchStackEntry* ss, i32 alpha, i32 beta, i32 depth, bool cutNode) {
        constexpr bool isRoot = NodeType == SearchNodeType::RootNode;
        constexpr bool isPV = NodeType != SearchNodeType::NonPVNode;

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

        const auto us = pos.ToMove;
        i32 score = -ScoreMate - MaxPly;
        i32 bestScore = -ScoreInfinite;

        const i32 startingAlpha = alpha;
        i32 probBeta;

        i16 rawEval = ScoreNone;
        i16 eval = ss->StaticEval;

        const bool doSkip = ss->Skip != Move::Null();
        bool improving = false;
        TTEntry _tte{};
        TTEntry* tte = &_tte;


        if (IsMain) {
            if ((++CheckupCount) >= CheckupMax) {
                CheckupCount = 0;

                if (CheckTime()) {
                    StopSearching = true;
                }
            }
            
            if ((Threads == 1 && Nodes >= MaxNodes)
                || (CheckupCount == 0 && Nodes >= MaxNodes)) {
                //  CheckupCount == 0 && thisThread.AssocPool.GetNodeCount() >= thisThread.AssocPool.SharedInfo.NodeLimit
                StopSearching = true;
            }
        }

        if (isPV) {
            SelDepth = std::max(SelDepth, ss->Ply + 1);
        }

        if (!isRoot) {
            if (pos.IsDraw()) {
                return MakeDrawScore(Nodes);
            }

            if (StopSearching || ss->Ply >= MaxSearchStackPly - 1) {
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

        const i16 ttScore = ss->TTHit ? MakeNormalScore(tte->Score(), ss->Ply) : ScoreNone;
        const Move ttMove = isRoot ? CurrentMove() : (ss->TTHit ? tte->BestMove : Move::Null());

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
            && !doSkip
            && depth >= NMPMinDepth
            && ss->Ply >= NMPPly
            && eval >= beta
            && eval >= ss->StaticEval
            && (ss - 1)->CurrentMove != Move::Null()
            && pos.HasNonPawnMaterial(us)) 
        {
            const auto reduction = NMPBaseRed + (depth / NMPDepthDiv) + std::min((eval - beta) / NMPEvalDiv, NMPEvalMin);
            ss->CurrentMove = Move::Null();
            ss->ContinuationHistory = &history->Continuations[0][0][0][0];

            pos.MakeNullMove();
            prefetch(TT.GetCluster(pos.Hash()));
            score = -Negamax<NonPVNode>(pos, ss + 1, -beta, -beta + 1, depth - reduction, !cutNode);
            pos.UnmakeNullMove();

            if (score >= beta) {

                if (NMPPly > 0 || depth <= 15)
                {
                    return score > ScoreWin ? beta : score;
                }

                NMPPly = (3 * (depth - reduction) / 4) + ss->Ply;
                i32 verification = Negamax<NonPVNode>(pos, ss, beta - 1, beta, depth - reduction, false);
                NMPPly = 0;

                if (verification >= beta)
                {
                    return score;
                }
            }
        }


        if (ttMove == Move::Null()) {
            if (cutNode && depth >= IIRMinDepth + 2)
                depth--;

            if (isPV && depth >= IIRMinDepth)
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
            ScoredMove captures[MoveListSize];
            i32 numCaps = GenerateQS(pos, captures, 0);
            AssignProbCutScores(pos, captures, numCaps);

            for (i32 i = 0; i < numCaps; i++) {
                Move m = OrderNextMove(captures, numCaps, i);
                if (!pos.IsLegal(m) || !pos.SEE_GE(m, std::max(1, probBeta - ss->StaticEval))) {
                    //  Skip illegal moves, and captures/promotions that don't result in a positive material trade
                    continue;
                }

                prefetch(TT.GetCluster(pos.HashAfter(m)));

                const auto [moveFrom, moveTo] = m.Unpack();
                const auto histIdx = MakePiece(us, bb.GetPieceAtIndex(moveFrom));

                ss->CurrentMove = m;
                ss->ContinuationHistory = &history->Continuations[0][1][histIdx][moveTo];
                Nodes++;

                pos.MakeMove(m);

                score = -QSearch<NonPVNode>(pos, ss + 1, -probBeta, -probBeta + 1);

                if (score >= probBeta) {
                    //  Verify at a low depth
                    score = -Negamax<NonPVNode>(pos, ss + 1, -probBeta, -probBeta + 1, depth - 3, !cutNode);
                }

                pos.UnmakeMove(m);

                if (score >= probBeta) {
                    tte->Update(pos.Hash(), MakeTTScore((short)score, ss->Ply), TTNodeType::Alpha, depth - 2, m, rawEval, ss->TTPV);
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


        i32 legalMoves = 0, playedMoves = 0;
        i32 quietCount = 0, captureCount = 0;

        bool didSkip = false;
        const auto lmpMoves = LMPTable[improving][depth];

        Move captureMoves[16];
        Move quietMoves[16];

        bool skipQuiets = false;

        ScoredMove list[MoveListSize];
        i32 size = Generate<PseudoLegal>(pos, list, 0);
        AssignScores(pos, ss, *history, list, size, ttMove);

        for (i32 i = 0; i < size; i++) {
            Move m = OrderNextMove(list, size, i);

            if (m == ss->Skip) {
                didSkip = true;
                continue;
            }

            if (!pos.IsLegal(m)) {
                continue;
            }

            const auto [moveFrom, moveTo] = m.Unpack();
            const auto theirPiece = bb.GetPieceAtIndex(moveTo);
            const auto ourPiece = bb.GetPieceAtIndex(moveFrom);
            const bool isCapture = pos.IsCapture(m);
            
            legalMoves++;
            i32 extend = 0;

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
                i32 singleBeta = ttScore - (SENumerator * depth / 10);
                i32 singleDepth = (depth + SEDepthAdj) / 2;

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

            const auto histIdx = MakePiece(us, ourPiece);

            ss->DoubleExtensions = static_cast<i16>((ss - 1)->DoubleExtensions + (extend >= 2 ? 1 : 0));
            ss->CurrentMove = m;
            ss->ContinuationHistory = &history->Continuations[ss->InCheck][isCapture][histIdx][moveTo];
            Nodes++;

            pos.MakeMove(m);

            playedMoves++;
            const u64 prevNodes = Nodes;

            if (isPV)
            {
                (ss + 1)->PVLength = 0;
            }

            i32 newDepth = depth + extend - 1;

            if (depth >= 2
                && legalMoves >= 2
                && !(isPV && isCapture))
            {

                i32 R = LogarithmicReductionTable[depth][legalMoves];

                R += (!improving);
                R += cutNode * 2;

                R -= ss->TTPV;
                R -= isPV;
                R -= (m == ss->KillerMove);

                i32 mHist = 2 * (isCapture ? history->CaptureHistory[us][ourPiece][moveTo][theirPiece] : history->MainHistory[us][m.GetMoveMask()]);    
                i32 histScore = mHist +
                                2 * (*(ss - 1)->ContinuationHistory)[histIdx][moveTo] +
                                    (*(ss - 2)->ContinuationHistory)[histIdx][moveTo] +
                                    (*(ss - 4)->ContinuationHistory)[histIdx][moveTo];

                R -= (histScore / (isCapture ? LMRCaptureDiv : LMRQuietDiv));

                i32 reducedDepth = std::max(0, std::min((newDepth - R), newDepth));

                score = -Negamax<NonPVNode>(pos, ss + 1, -alpha - 1, -alpha, reducedDepth, true);

                if (score > alpha && reducedDepth < newDepth) {
                    bool deeper    = score > (bestScore + LMRExtMargin + 2 * newDepth);
                    bool shallower = score < (bestScore + newDepth);

                    newDepth += deeper - shallower;

                    if (newDepth > reducedDepth) {
                        score = -Negamax<NonPVNode>(pos, ss + 1, -alpha - 1, -alpha, newDepth, !cutNode);
                    }

                    i32 bonus = score <= alpha ? -StatBonus(newDepth) :
                                score >= beta  ?  StatBonus(newDepth) :
                                                  0;

                    UpdateContinuations(ss, us, ourPiece, m.To(), bonus);
                }
            }
            else if (!isPV || legalMoves > 1) {
                score = -Negamax<NonPVNode>(pos, ss + 1, -alpha - 1, -alpha, newDepth, !cutNode);
            }

            if (isPV && (playedMoves == 1 || score > alpha)) {
                (ss + 1)->PV[0] = Move::Null();
                (ss + 1)->PVLength = 0;
                score = -Negamax<PVNode>(pos, ss + 1, -beta, -alpha, newDepth, false);
            }

            pos.UnmakeMove(m);

            if (isRoot) {
                NodeTable[moveFrom][moveTo] += Nodes - prevNodes;
            }

            if (StopSearching) {
                return ScoreDraw;
            }

            if (isRoot) {
                i32 rmIndex = 0;
                for (i32 j = 0; j < RootMoves.size(); j++) {
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
            
            tte->Update(pos.Hash(), MakeTTScore(static_cast<i16>(bestScore), ss->Ply), bound, depth, moveToSave, rawEval, ss->TTPV);

            if (!ss->InCheck
                && (bestMove.IsNull() || !pos.IsNoisy(bestMove))
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
    i32 SearchThread::QSearch(Position& pos, SearchStackEntry* ss, i32 alpha, i32 beta) {
        
        constexpr bool isPV = NodeType != SearchNodeType::NonPVNode;

        if (alpha < ScoreDraw && pos.HasCycle(ss->Ply))
        {
            alpha = MakeDrawScore(Nodes);
            if (alpha >= beta)
                return alpha;
        }

        Bitboard& bb = pos.bb;
        HistoryTable* history = &History;

        Move bestMove = Move::Null();

        const auto us = pos.ToMove;
        i32 score = -ScoreMate - MaxPly;
        i32 bestScore = -ScoreInfinite;
        i32 futility = -ScoreInfinite;

        i16 rawEval = ScoreNone;
        i16 eval = ss->StaticEval;

        TTEntry _tte{};
        TTEntry* tte = &_tte;
        ss->InCheck = pos.Checked();
        ss->TTHit = TT.Probe(pos.Hash(), tte);
        const i16 ttScore = ss->TTHit ? MakeNormalScore(tte->Score(), ss->Ply) : ScoreNone;
        const Move ttMove = ss->TTHit ? tte->BestMove : Move::Null();
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
                rawEval = ((ss - 1)->CurrentMove == Move::Null()) ? (-(ss - 1)->StaticEval) : NNUE::GetEvaluation(pos);

                eval = ss->StaticEval = AdjustEval(pos, us, rawEval);
            }

            if (eval >= beta) {
                if (!ss->TTHit)
                    tte->Update(pos.Hash(), MakeTTScore(eval, ss->Ply), TTNodeType::Alpha, TTEntry::DepthNone, Move::Null(), rawEval, false);

                if (std::abs(eval) < ScoreTTWin)
                    eval = static_cast<i16>((4 * eval + beta) / 5);

                return eval;
            }

            if (eval > alpha) {
                alpha = eval;
            }

            bestScore = eval;

            futility = (std::min(ss->StaticEval, static_cast<i16>(bestScore)) + QSFutileMargin);
        }

        const auto prevSquare = (ss - 1)->CurrentMove == Move::Null() ? SQUARE_NB : (ss - 1)->CurrentMove.To();
        i32 legalMoves = 0;
        i32 checkEvasions = 0;

        ScoredMove list[MoveListSize];
        i32 size = GenerateQS(pos, list, 0);
        AssignQuiescenceScores(pos, ss, *history, list, size, ttMove);

        for (i32 i = 0; i < size; i++)
        {
            Move m = OrderNextMove(list, size, i);

            if (!pos.IsLegal(m))
            {
                continue;
            }

            legalMoves++;

            const auto [moveFrom, moveTo] = m.Unpack();
            const auto theirPiece = bb.GetPieceAtIndex(moveTo);
            const auto ourPiece = bb.GetPieceAtIndex(moveFrom);

            const bool isCapture = pos.IsCapture(m);
            const bool givesCheck = ((pos.State->CheckSquares[ourPiece] & SquareBB(moveTo)) != 0);

            if (bestScore > ScoreTTLoss) {
                if (!(givesCheck || m.IsPromotion())
                    && (prevSquare != moveTo)
                    && futility > -ScoreWin) {

                    if (legalMoves > 3 && !ss->InCheck) {
                        continue;
                    }

                    i32 futilityValue = (futility + GetPieceValue(theirPiece));

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

            ss->CurrentMove = m;
            ss->ContinuationHistory = &History.Continuations[ss->InCheck][isCapture][MakePiece(us, ourPiece)][moveTo];
            Nodes++;

            pos.MakeMove(m);
            score = -QSearch<NodeType>(pos, ss + 1, -beta, -alpha);
            pos.UnmakeMove(m);

            if (score > bestScore) {
                bestScore = static_cast<i16>(score);

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

        tte->Update(pos.Hash(), MakeTTScore(static_cast<i16>(bestScore), ss->Ply), bound, 0, bestMove, rawEval, ss->TTPV);

        return bestScore;
    }




    void SearchThread::UpdatePV(Move* pv, Move move, Move* childPV) {
        for (*pv++ = move; childPV != nullptr && *childPV != Move::Null();) {
            *pv++ = *childPV++;
        }
        *pv = Move::Null();
    }


    void SearchThread::UpdateStats(Position& pos, SearchStackEntry* ss, Move bestMove, i32 bestScore, i32 beta, i32 depth, 
                                   Move* quietMoves, i32 quietCount, Move* captureMoves, i32 captureCount) {

        HistoryTable& history = this->History;
        const auto [bmFrom, bmTo] = bestMove.Unpack();

        Bitboard& bb = pos.bb;

        const auto us = pos.ToMove;
        const auto bmPiece = bb.GetPieceAtIndex(bmFrom);
        const auto bmCapPiece = bb.GetPieceAtIndex(bmTo);

        const auto bonus = StatBonus(depth);
        const auto malus = StatMalus(depth);

        if (bmCapPiece != Piece::NONE && !bestMove.IsCastle())
        {
            history.CaptureHistory[us][bmPiece][bmTo][bmCapPiece] << bonus;
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
                return;

            history.MainHistory[us][bestMove.GetMoveMask()] << bonus;
            if (ss->Ply < LowPlyCount)
                history.PlyHistory[ss->Ply][bestMove.GetMoveMask()] << bonus;

            UpdateContinuations(ss, us, bmPiece, bmTo, bonus);


            for (i32 i = 0; i < quietCount; i++) {
                Move m = quietMoves[i];
                const auto [moveFrom, moveTo] = m.Unpack();
                const auto thisPiece = bb.GetPieceAtIndex(moveFrom);

                history.MainHistory[us][m.GetMoveMask()] << -malus;
                if (ss->Ply < LowPlyCount)
                    history.PlyHistory[ss->Ply][m.GetMoveMask()] << -malus;

                UpdateContinuations(ss, us, thisPiece, moveTo, -malus);
            }
        }

        for (i32 i = 0; i < captureCount; i++) {
            Move m = captureMoves[i];
            const auto [moveFrom, moveTo] = m.Unpack();
            const auto thisPiece = bb.GetPieceAtIndex(moveFrom);
            const auto capturedPiece = bb.GetPieceAtIndex(moveTo);

            history.CaptureHistory[us][thisPiece][moveTo][capturedPiece] << -malus;
        }

    }


    i16 SearchThread::AdjustEval(Position& pos, i32 us, i16 rawEval) {
        rawEval = static_cast<i16>(rawEval * (200 - pos.State->HalfmoveClock) / 200);

        const auto pawn = History.PawnCorrection[us][pos.PawnHash() % 16384] / CorrectionGrain;
        const auto nonPawnW = History.NonPawnCorrection[us][pos.NonPawnHash(Color::WHITE) % 16384] / CorrectionGrain;
        const auto nonPawnB = History.NonPawnCorrection[us][pos.NonPawnHash(Color::BLACK) % 16384] / CorrectionGrain;
        const auto corr = (pawn * 200 + nonPawnW * 100 + nonPawnB * 100) / 300;

        return static_cast<i16>(rawEval + corr);
    }


    void SearchThread::UpdateCorrectionHistory(Position& pos, i32 diff, i32 depth) {

        const auto scaledWeight = std::min((depth * depth) + 1, 128);

        auto& pawnCh = History.PawnCorrection[pos.ToMove][pos.PawnHash() % 16384];
        const auto pawnBonus = (pawnCh * (CorrectionScale - scaledWeight) + (diff * CorrectionGrain * scaledWeight)) / CorrectionScale;
        pawnCh = std::clamp(pawnBonus, -CorrectionMax, CorrectionMax);

        auto& nonPawnChW = History.NonPawnCorrection[pos.ToMove][pos.NonPawnHash(Color::WHITE) % 16384];
        const auto nonPawnBonusW = (nonPawnChW * (CorrectionScale - scaledWeight) + (diff * CorrectionGrain * scaledWeight)) / CorrectionScale;
        nonPawnChW = std::clamp(nonPawnBonusW, -CorrectionMax, CorrectionMax);

        auto& nonPawnChB = History.NonPawnCorrection[pos.ToMove][pos.NonPawnHash(Color::BLACK) % 16384];
        const auto nonPawnBonusB = (nonPawnChB * (CorrectionScale - scaledWeight) + (diff * CorrectionGrain * scaledWeight)) / CorrectionScale;
        nonPawnChB = std::clamp(nonPawnBonusB, -CorrectionMax, CorrectionMax);
    }


    void SearchThread::UpdateContinuations(SearchStackEntry* ss, i32 pc, i32 pt, i32 sq, i32 bonus) {
        const auto piece = MakePiece(pc, pt);

        for (auto i : {1, 2, 4, 6}) {
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





    Move SearchThread::OrderNextMove(ScoredMove* moves, i32 size, i32 listIndex) {
        i32 max = INT32_MIN;
        i32 maxIndex = listIndex;

        for (i32 i = listIndex; i < size; i++) {
            if (moves[i].Score > max) {
                max = moves[i].Score;
                maxIndex = i;
            }
        }

        std::swap(moves[maxIndex], moves[listIndex]);

        return moves[listIndex].move;
    }



    void SearchThread::AssignProbCutScores(Position& pos, ScoredMove* list, i32 size) {
        
        Bitboard& bb = pos.bb;
        for (i32 i = 0; i < size; i++) {
            Move m = list[i].move;

            list[i].Score = GetSEEValue(m.IsEnPassant() ? PAWN : bb.GetPieceAtIndex(m.To()));
            if (m.IsPromotion()) {
                list[i].Score += GetSEEValue(QUEEN) + 1;
            }
        }
    }



    void SearchThread::AssignQuiescenceScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, i32 size, Move ttMove) {
        Bitboard& bb = pos.bb;
        const auto pc = pos.ToMove;

        for (i32 i = 0; i < size; i++) {
            Move m = list[i].move;
            const auto [moveFrom, moveTo] = m.Unpack();
            const auto pt = bb.GetPieceAtIndex(moveFrom);

            if (m == ttMove) {
                list[i].Score = INT32_MAX - 100000;
            }
            else if (bb.GetPieceAtIndex(moveTo) != Piece::NONE && !m.IsCastle()) {
                const auto capturedPiece = bb.GetPieceAtIndex(moveTo);
                auto& hist = history.CaptureHistory[pc][pt][moveTo][capturedPiece];
                list[i].Score = (MVVMult * GetPieceValue(capturedPiece)) + hist;
            }
            else {
                i32 contIdx = MakePiece(pc, pt);

                list[i].Score =  2 * history.MainHistory[pc][m.GetMoveMask()];
                list[i].Score += 2 * (*(ss - 1)->ContinuationHistory)[contIdx][moveTo];
                list[i].Score +=     (*(ss - 2)->ContinuationHistory)[contIdx][moveTo];
                list[i].Score +=     (*(ss - 4)->ContinuationHistory)[contIdx][moveTo];
                list[i].Score +=     (*(ss - 6)->ContinuationHistory)[contIdx][moveTo];

                if ((pos.State->CheckSquares[pt] & SquareBB(moveTo)) != 0) {
                    list[i].Score += CheckBonus;
                }
            }

            if (pt == HORSIE) {
                list[i].Score += 200;
            }
        }
    }



    void SearchThread::AssignScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, i32 size, Move ttMove) {
        
        Bitboard& bb = pos.bb;
        const auto pc = pos.ToMove;

        for (i32 i = 0; i < size; i++) {
            Move m = list[i].move;
            const auto [moveFrom, moveTo] = m.Unpack();
            const auto pt = bb.GetPieceAtIndex(moveFrom);

            if (m == ttMove) {
                list[i].Score = INT32_MAX - 100000;
            }
            else if (m == ss->KillerMove) {
                list[i].Score = INT32_MAX - 1000000;
            }
            else if (bb.GetPieceAtIndex(moveTo) != Piece::NONE && !m.IsCastle()) {
                const auto capturedPiece = bb.GetPieceAtIndex(moveTo);
                auto& hist = history.CaptureHistory[pc][pt][moveTo][capturedPiece];
                list[i].Score = (MVVMult * GetPieceValue(capturedPiece)) + hist;
            }
            else {
                i32 contIdx = MakePiece(pc, pt);

                list[i].Score =  2 * history.MainHistory[pc][m.GetMoveMask()];
                list[i].Score += 2 * (*(ss - 1)->ContinuationHistory)[contIdx][moveTo];
                list[i].Score +=     (*(ss - 2)->ContinuationHistory)[contIdx][moveTo];
                list[i].Score +=     (*(ss - 4)->ContinuationHistory)[contIdx][moveTo];
                list[i].Score +=     (*(ss - 6)->ContinuationHistory)[contIdx][moveTo];

                if (ss->Ply < LowPlyCount)
                {
                    list[i].Score += ((2 * LowPlyCount + 1) * history.PlyHistory[ss->Ply][m.GetMoveMask()]) / (2 * ss->Ply + 1);
                }

                if ((pos.State->CheckSquares[pt] & SquareBB(moveTo)) != 0) {
                    list[i].Score += CheckBonus;
                }
            }

            if (pt == HORSIE) {
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