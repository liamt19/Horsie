

#include "search.h"

#include "movegen.h"
#include "nnue/nn.h"
#include "position.h"
#include "precomputed.h"
#include "search_options.h"
#include "threadpool.h"
#include "tt.h"
#include "util/dbg_hit.h"
#include "util/timer.h"
#include "wdl.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>

using namespace Horsie::Search;

namespace Horsie {

    constexpr i32 MaxSearchStackPly = 256 - 10;
    constexpr i32 BoundLower = static_cast<i32>(TTNodeType::Alpha);
    constexpr i32 BoundUpper = static_cast<i32>(TTNodeType::Beta);

    constexpr double StabilityCoefficients[] = { 2.2, 1.6, 1.4, 1.1, 1, 0.95, 0.9 };
    constexpr i32 StabilityMax = 6;

    void SearchThread::MainThreadSearch() {
        TT->TTUpdate();

        AssocPool->AwakenHelperThreads();
        this->Search(AssocPool->SharedInfo);

        while (!ShouldStop() && AssocPool->SharedInfo.IsInfinite()) {}

        AssocPool->StopAllThreads();
        AssocPool->WaitForSearchFinished();

        if (OnSearchFinish) {
            OnSearchFinish();
        }
    }

    void SearchThread::Search(SearchLimits& info) {
        HardNodeLimit = info.MaxNodes;
        HardTimeLimit = info.MaxSearchTime;

        SearchStackEntry _SearchStackBlock[MaxPly] = {};
        SearchStackEntry* ss = &_SearchStackBlock[10];
        for (i32 i = -10; i < MaxSearchStackPly; i++) {
            (ss + i)->Clear();
            (ss + i)->Ply = static_cast<i16>(i);
            (ss + i)->PV = AlignedAlloc<Move>(MaxPly);
            (ss + i)->PVLength = 0;
            (ss + i)->ContinuationHistory = &History.Continuations[0][0][0][0];
        }

        NNUE::ResetCaches(RootPosition);

        i32 multiPV = std::min(i32(MultiPV), i32(RootMoves.size()));

        std::vector<i32> searchScores(MaxPly);

        RootMove lastBestRootMove = RootMove(Move::Null());
        i32 stability = 0;

        i32 maxDepth = IsMain() ? MaxDepth : MaxPly;
        while (++RootDepth < maxDepth) {
            //  The main thread is not allowed to search past info.MaxDepth
            if (IsMain() && RootDepth > info.MaxDepth)
                break;

            if (ShouldStop())
                break;

            for (RootMove& rm : RootMoves) {
                rm.PreviousScore = rm.Score;
            }

            i32 usedDepth = RootDepth;

            for (PVIndex = 0; PVIndex < multiPV; PVIndex++) {
                if (ShouldStop())
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
                    score = Negamax<RootNode>(RootPosition, ss, alpha, beta, std::max(1, usedDepth), false);

                    std::stable_sort(RootMoves.begin() + PVIndex, RootMoves.end());

                    if (ShouldStop())
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

                std::stable_sort(RootMoves.begin() + 0, RootMoves.end());

                if (IsMain() && (ShouldStop() || PVIndex == multiPV - 1)) {
                    if (OnDepthFinish) {
                        OnDepthFinish();
                    }
                }
            }

            if (!IsMain())
                continue;

            if (ShouldStop()) {

                //  If we received a stop command or hit the hard time limit, our RootMoves may not have been filled in properly.
                //  In that case, we replace the current bestmove with the last depth's bestmove
                //  so that the move we send is based on an entire depth being searched instead of only a portion of it.
                RootMoves[0] = lastBestRootMove;

                for (i32 i = -10; i < MaxSearchStackPly; i++) {
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
            for (i32 i = 0; i < RootMoves[0].PV.size(); i++) {
                lastBestRootMove.PV.push_back(RootMoves[0].PV[i]);
                if (lastBestRootMove.PV[i] == Move::Null()) {
                    break;
                }
            }

            searchScores.push_back(RootMoves[0].Score);

            if (HasSoftTime()) {
                //  Base values taken from Clarity
                double multFactor = 1.0;
                if (RootDepth > 7) {

                    double nodeTM = (1.5 - RootMoves[0].Nodes / static_cast<double>(Nodes)) * 1.75;
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

            if (Nodes >= info.SoftNodeLimit) {
                break;
            }

            if (!ShouldStop()) {
                CompletedDepth = RootDepth;
            }
        }

        if (IsMain() && RootDepth >= MaxDepth && Nodes != HardNodeLimit && !ShouldStop()) {
            SetStop();
        }

        for (i32 i = -10; i < MaxSearchStackPly; i++) {
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

        if (!isRoot && alpha < ScoreDraw && pos.HasCycle(ss->Ply)) {
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

        const i32 startAlpha = alpha;
        i32 probBeta;

        i16 rawEval = ScoreNone;
        i16 eval = ss->StaticEval;

        const bool doSkip = ss->Skip != Move::Null();
        bool improving = false;
        TTEntry _tte{};
        TTEntry* tte = &_tte;

        if (IsMain()) {
            CheckLimits();
        }

        if (isPV) {
            SelDepth = std::max(SelDepth, ss->Ply + 1);
        }

        if (!isRoot) {
            if (pos.IsDraw()) {
                return MakeDrawScore(Nodes);
            }

            if (ShouldStop() || ss->Ply >= MaxSearchStackPly - 1) {
                if (pos.InCheck()) {
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
        ss->InCheck = pos.InCheck();
        ss->TTHit = TT->Probe(pos.Hash(), tte);
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
            && (tte->Bound() & (ttScore >= beta ? BoundLower : BoundUpper)) != 0) {
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

            tte->Update(pos.Hash(), ScoreNone, TTNodeType::Invalid, TTEntry::DepthNone, Move::Null(), rawEval, TT->Age, ss->TTPV);
        }

        if (ss->Ply >= 2) {
            improving = (ss - 2)->StaticEval != ScoreNone ? ss->StaticEval > (ss - 2)->StaticEval :
                       ((ss - 4)->StaticEval != ScoreNone ? ss->StaticEval > (ss - 4)->StaticEval : true);
        }

        if (!(ss - 1)->InCheck 
            && (ss - 1)->CurrentMove != Move::Null() 
            && pos.CapturedPiece() == NONE) {

            const i32 val = -QuietOrderMult * ((ss - 1)->StaticEval + ss->StaticEval);
            const auto bonus = std::clamp(val, -QuietOrderMin, (i32)QuietOrderMax);
            history->MainHistory[Not(us)][(ss - 1)->CurrentMove.GetMoveMask()] << bonus;
        }


        if (UseRFP
            && !ss->TTPV
            && !doSkip
            && depth <= RFPMaxDepth
            && ttMove == Move::Null()
            && (eval < ScoreAssuredWin)
            && (eval >= beta)
            && (eval - GetRFPMargin(depth, improving)) >= beta) {
            
            return (eval + beta) / 2;
        }


        if (UseRazoring
            && !isPV
            && !doSkip
            && depth <= RazoringMaxDepth
            && alpha < 2000
            && ss->StaticEval + RazoringMult * depth <= alpha) {

            score = QSearch<NodeType>(pos, ss, alpha, alpha + 1);
            if (score <= alpha)
                return score;
        }


        probBeta = beta + (improving ? ProbcutBetaImp : ProbcutBeta);
        if (UseProbcut
            && !isPV
            && !doSkip
            && std::abs(beta) < ScoreTTWin
            && (!ss->TTHit || tte->Depth() < depth - 3 || tte->Score() >= probBeta)) {

            ScoredMove captures[MoveListSize];
            i32 numCaps = GenerateQS(pos, captures, 0);
            AssignProbcutScores(pos, captures, numCaps);

            for (i32 i = 0; i < numCaps; i++) {
                Move m = OrderNextMove(captures, numCaps, i);
                if (!pos.IsLegal(m) || !pos.SEE_GE(m, std::max(1, probBeta - ss->StaticEval))) {
                    continue;
                }

                prefetch(TT->GetCluster(pos.HashAfter(m)));

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
                    tte->Update(pos.Hash(), MakeTTScore(static_cast<i16>(score), ss->Ply), TTNodeType::Alpha, depth - 2, m, rawEval, TT->Age, ss->TTPV);
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


        probBeta = beta + (improving ? ProbcutBetaImp : ProbcutBeta);
        if (UseProbcut
            && !isPV
            && !doSkip
            && std::abs(beta) < ScoreTTWin
            && (!ss->TTHit || tte->Depth() < depth - 3 || tte->Score() >= probBeta)) {

            ScoredMove captures[MoveListSize];
            i32 numCaps = GenerateQS(pos, captures, 0);
            AssignProbcutScores(pos, captures, numCaps);

            for (i32 i = 0; i < numCaps; i++) {
                Move m = OrderNextMove(captures, numCaps, i);
                if (!pos.IsLegal(m) || !pos.SEE_GE(m, std::max(1, probBeta - ss->StaticEval))) {
                    //  Skip illegal moves, and captures/promotions that don't result in a positive material trade
                    continue;
                }

                prefetch(TT->GetCluster(pos.HashAfter(m)));

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
                    tte->Update(pos.Hash(), MakeTTScore(static_cast<i16>(score), ss->Ply), TTNodeType::Alpha, depth - 2, m, rawEval, TT->Age, ss->TTPV);
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
            && std::abs(beta) < ScoreTTWin) {
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

            if (isRoot && MultiPV != 1) {
                //  If this move is ordered before PVIndex, it's already been searched.
                if (std::count(RootMoves.begin() + PVIndex, RootMoves.end(), m) == 0)
                    continue;
            }

            const auto [moveFrom, moveTo] = m.Unpack();
            const auto theirPiece = bb.GetPieceAtIndex(moveTo);
            const auto ourPiece = bb.GetPieceAtIndex(moveFrom);
            const bool isCapture = pos.IsCapture(m);
            
            legalMoves++;
            i32 extend = 0;
            i32 R = LogarithmicReductionTable[depth][legalMoves];

            i32 moveHist = isCapture ? history->CaptureHistory[us][ourPiece][moveTo][theirPiece] : history->MainHistory[us][m.GetMoveMask()];

            if (ShallowPruning
                && !isRoot
                && bestScore > ScoreMatedMax
                && pos.HasNonPawnMaterial(us)) {

                if (skipQuiets == false)
                    skipQuiets = legalMoves >= lmpMoves;

                const bool givesCheck = pos.GivesCheck(ourPiece, moveTo);
                const bool isQuiet = !(givesCheck || isCapture);

                if (isQuiet && skipQuiets && depth <= ShallowMaxDepth)
                    continue;

                i32 lmrRed = (R * 1024) + NMFutileBase;

                lmrRed += !isPV * NMFutilePVCoeff;
                lmrRed += !improving * NMFutileImpCoeff;
                lmrRed -= (moveHist / (isCapture ? LMRCaptureDiv : LMRQuietDiv)) * NMFutileHistCoeff;

                lmrRed /= 1024;
                i32 lmrDepth = std::max(0, depth - lmrRed);

                i32 futilityMargin = NMFutMarginB + (lmrDepth * NMFutMarginM) + (moveHist / NMFutMarginDiv);
                if (isQuiet 
                    && !ss->InCheck
                    && lmrDepth <= 8 
                    && ss->StaticEval + futilityMargin < alpha) {
                    skipQuiets = true;
                    continue;
                }

                const auto seeMargin = -ShallowSEEMargin * depth;
                if ((!isQuiet || skipQuiets) && !pos.SEE_GE(m, seeMargin)) {
                    continue;
                }
            }


            if (UseSingularExtensions
                && !isRoot
                && !doSkip
                && m == ttMove
                && ss->Ply < RootDepth * 2
                && std::abs(ttScore) < ScoreWin) {

                const auto SEDepth = SEMinDepth + (isPV && tte->PV());

                if (depth >= SEDepth
                    && ((tte->Bound() & BoundLower) != 0)
                    && tte->Depth() >= depth - 3) {

                    i32 singleBeta = ttScore - (SENumerator * depth / 10);
                    i32 singleDepth = (depth + SEDepthAdj) / 2;

                    ss->Skip = m;
                    score = Negamax<NonPVNode>(pos, ss, singleBeta - 1, singleBeta, singleDepth, cutNode);
                    ss->Skip = Move::Null();

                    if (score < singleBeta) {
                        bool doubleExt = !isPV && ss->DoubleExtensions <= 8 && (score < singleBeta - SEDoubleMargin);
                        bool tripleExt = doubleExt && (score < singleBeta - SETripleMargin - (isCapture * SETripleCapSub));

                        extend = 1 + doubleExt + tripleExt;
                    }
                    else if (singleBeta >= beta) {
                        return singleBeta;
                    }
                    else if (ttScore >= beta) {
                        extend = -2 + isPV;
                    }
                    else if (cutNode) {
                        extend = -2;
                    }
                    else if (ttScore <= alpha) {
                        extend = -1;
                    }
                }
                else if (depth < SEDepth
                         && !ss->InCheck
                         && eval < alpha - 25
                         && tte->Bound() & BoundLower) {
                    
                    extend = 1;
                }
            }

            prefetch(TT->GetCluster(pos.HashAfter(m)));

            const auto histIdx = MakePiece(us, ourPiece);

            ss->DoubleExtensions = static_cast<i16>((ss - 1)->DoubleExtensions + (extend >= 2 ? 1 : 0));
            ss->CurrentMove = m;
            ss->ContinuationHistory = &history->Continuations[ss->InCheck][isCapture][histIdx][moveTo];
            Nodes++;

            pos.MakeMove(m);

            playedMoves++;
            const u64 prevNodes = Nodes;

            if (isPV) {
                (ss + 1)->PVLength = 0;
            }

            i32 newDepth = depth + extend;

            if (depth >= 2
                && legalMoves >= 2
                && !(isPV && isCapture)) {

                R += (!improving);
                R += cutNode * 2;

                R -= ss->TTPV;
                R -= isPV;
                R -= (m == ss->KillerMove);

                i32 histScore = 2 * moveHist +
                                2 * (*(ss - 1)->ContinuationHistory)[histIdx][moveTo] +
                                    (*(ss - 2)->ContinuationHistory)[histIdx][moveTo] +
                                    (*(ss - 4)->ContinuationHistory)[histIdx][moveTo];

                R -= (histScore / (isCapture ? LMRCaptureDiv : LMRQuietDiv));

                R = std::max(1, std::min(R, newDepth));
                i32 reducedDepth = (newDepth - R);

                score = -Negamax<NonPVNode>(pos, ss + 1, -alpha - 1, -alpha, reducedDepth, true);

                if (score > alpha && R > 1) {
                    newDepth += (score > (bestScore + DeeperMargin + 4 * newDepth));
                    newDepth -= (score < (bestScore + newDepth));

                    if (newDepth - 1 > reducedDepth) {
                        score = -Negamax<NonPVNode>(pos, ss + 1, -alpha - 1, -alpha, newDepth - 1, !cutNode);
                    }

                    i32 bonus = 0;
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

            if (ShouldStop()) {
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
                rm.Nodes += (Nodes - prevNodes);

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
            TTNodeType bound = bestScore >= beta      ? TTNodeType::Alpha
                             : bestScore > startAlpha ? TTNodeType::Exact
                             :                          TTNodeType::Beta;

            Move moveToSave = (bound == TTNodeType::Beta) ? Move::Null() : bestMove;
            
            tte->Update(pos.Hash(), MakeTTScore(static_cast<i16>(bestScore), ss->Ply), bound, depth, moveToSave, rawEval, TT->Age, ss->TTPV);

            if (!ss->InCheck
                && (bestMove.IsNull() || !pos.IsNoisy(bestMove))
                && !(bound == TTNodeType::Alpha && bestScore <= ss->StaticEval)
                && !(bound == TTNodeType::Beta && bestScore >= ss->StaticEval)) {

                auto diff = bestScore - ss->StaticEval;
                UpdateCorrectionHistory(pos, diff, depth);
            }
        }


        return bestScore;
    }


    template <SearchNodeType NodeType>
    i32 SearchThread::QSearch(Position& pos, SearchStackEntry* ss, i32 alpha, i32 beta) {
        constexpr bool isPV = NodeType != SearchNodeType::NonPVNode;

        if (alpha < ScoreDraw && pos.HasCycle(ss->Ply)) {
            alpha = MakeDrawScore(Nodes);
            if (alpha >= beta)
                return alpha;
        }

        Bitboard& bb = pos.bb;
        HistoryTable* history = &History;

        Move bestMove = Move::Null();

        const auto us = pos.ToMove;
        const bool inCheck = pos.InCheck();
        i32 score = -ScoreMate - MaxPly;
        i32 bestScore = -ScoreInfinite;
        i32 futility = -ScoreInfinite;

        i16 rawEval = ScoreNone;
        i16 eval = ss->StaticEval;

        TTEntry _tte{};
        TTEntry* tte = &_tte;
        ss->InCheck = inCheck;
        ss->TTHit = TT->Probe(pos.Hash(), tte);
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
            return inCheck ? ScoreDraw : NNUE::GetEvaluation(pos);
        }

        if (!isPV
            && ttScore != ScoreNone
            && (tte->Bound() & (ttScore >= beta ? BoundLower : BoundUpper)) != 0) {
            return ttScore;
        }

        if (inCheck) {
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
                    tte->Update(pos.Hash(), MakeTTScore(eval, ss->Ply), TTNodeType::Alpha, TTEntry::DepthNone, Move::Null(), rawEval, TT->Age, false);

                if (std::abs(eval) < ScoreTTWin)
                    eval = static_cast<i16>((4 * eval + beta) / 5);

                return eval;
            }

            alpha = std::max(static_cast<i32>(eval), alpha);

            bestScore = eval;
            futility = bestScore + QSFutileMargin;
        }

        i32 legalMoves = 0;
        i32 quietEvasions = 0;

        ScoredMove list[MoveListSize];
        i32 size = GenerateQS(pos, list, 0);
        AssignQuiescenceScores(pos, ss, *history, list, size, ttMove);

        for (i32 i = 0; i < size; i++) {
            Move m = OrderNextMove(list, size, i);

            if (!pos.IsLegal(m)) {
                continue;
            }

            legalMoves++;

            const auto [moveFrom, moveTo] = m.Unpack();
            const auto ourPiece = bb.GetPieceAtIndex(moveFrom);

            const bool isCapture = pos.IsCapture(m);
            if (bestScore > ScoreTTLoss) {

                if (quietEvasions >= 2)
                    break;

                if (!inCheck && legalMoves > 3)
                    continue;

                if (!inCheck && futility <= alpha && !pos.SEE_GE(m, 1)) {
                    bestScore = std::max(bestScore, futility);
                    continue;
                }

                if (!inCheck && !pos.SEE_GE(m, -QSSeeMargin)) {
                    continue;
                }
            }

            prefetch(TT->GetCluster(pos.HashAfter(m)));

            if (inCheck && !isCapture) {
                quietEvasions++;
            }

            ss->CurrentMove = m;
            ss->ContinuationHistory = &History.Continuations[inCheck][isCapture][MakePiece(us, ourPiece)][moveTo];
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

        if (inCheck && legalMoves == 0)
            return MakeMateScore(ss->Ply);

        TTNodeType bound = (bestScore >= beta) ? TTNodeType::Alpha : TTNodeType::Beta;

        tte->Update(pos.Hash(), MakeTTScore(static_cast<i16>(bestScore), ss->Ply), bound, 0, bestMove, rawEval, TT->Age, ttPV);

        return bestScore;
    }

    void SearchThread::UpdatePV(Move* pv, Move move, Move* childPV) const {
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

        if (bmCapPiece != Piece::NONE && !bestMove.IsCastle()) {
            history.CaptureHistory[us][bmPiece][bmTo][bmCapPiece] << bonus;
        }
        else {

            if (!bestMove.IsEnPassant()) {
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

    i16 SearchThread::AdjustEval(Position& pos, i32 us, i16 rawEval) const {
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

    void SearchThread::UpdateContinuations(SearchStackEntry* ss, i32 pc, i32 pt, i32 sq, i32 bonus) const {
        const auto piece = MakePiece(pc, pt);

        for (auto i : {1, 2, 4, 6}) {
            if (ss->InCheck && i > 2) {
                break;
            }

            if ((ss - i)->CurrentMove != Move::Null()) {
                (*(ss - i)->ContinuationHistory)[piece][sq] << bonus;
            }
        }
    }

    void SearchThread::PrintSearchInfo() const {

        const auto nodes = AssocPool->GetNodeCount();
        const auto duration = Timepoint::TimeSince(StartTime);
        const auto durCnt = std::max(1.0, static_cast<double>(duration));
        const auto nodesPerSec = static_cast<u64>(nodes / (durCnt / 1000));
        const auto hashfull = TT->Hashfull();
        
        const auto mpv = std::min(i32(MultiPV), i32(RootMoves.size()));
        for (size_t i = 0; i < mpv; i++) {
            const RootMove& rm = RootMoves[i];

            bool moveSearched = rm.Score != -ScoreInfinite;
            i32 depth = moveSearched ? RootDepth : std::max(1, RootDepth - 1);
            i32 moveScore = moveSearched ? rm.Score : rm.PreviousScore;

            std::cout << "info depth " << depth;
            std::cout << " seldepth " << rm.Depth;
            std::cout << " multipv " << (i + 1);
            std::cout << " time " << durCnt;
            std::cout << " score " << FormatMoveScore(WDL::NormalizeScore(moveScore));

            if (UCI_ShowWDL) {
                if (moveScore > ScoreWin) {
                    std::cout << " wdl 1000 0 0";
                }
                else if (moveScore < -ScoreWin) {
                    std::cout << " wdl 0 0 1000";
                }
                else {
                    const auto material = RootPosition.MaterialCount();
                    const auto [win, loss] = WDL::MaterialModel(moveScore, material);
                    const auto draw = 1000 - win - loss;
                    std::cout << " wdl " << win << " " << draw << " " << loss;
                }
            }

            std::cout << " nodes " << nodes;
            std::cout << " nps " << nodesPerSec;
            std::cout << " hashfull " << hashfull;
            std::cout << " pv";

            for (i32 j = 0; j < rm.PV.size(); j++) {
                if (rm.PV[j] == Move::Null())
                    break;

                std::cout << " " << rm.PV[j].SmithNotation(RootPosition.IsChess960);
            }

            std::cout << std::endl;
        }

    }

    Move SearchThread::OrderNextMove(ScoredMove* moves, i32 size, i32 listIndex) const {
        i32 max = INT32_MIN;
        i32 maxIndex = listIndex;

        for (i32 i = listIndex; i < size; i++) {
            if (moves[i].score > max) {
                max = moves[i].score;
                maxIndex = i;
            }
        }

        std::swap(moves[maxIndex], moves[listIndex]);

        return moves[listIndex].move;
    }

    void SearchThread::AssignProbcutScores(Position& pos, ScoredMove* list, i32 size) const {
        Bitboard& bb = pos.bb;
        for (i32 i = 0; i < size; i++) {
            Move m = list[i].move;

            list[i].score = m.IsEnPassant() ? PAWN : bb.GetPieceAtIndex(m.To());
            if (m.IsPromotion()) {
                list[i].score += 10;
            }
        }
    }

    void SearchThread::AssignQuiescenceScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, i32 size, Move ttMove) const {
        Bitboard& bb = pos.bb;
        const auto pc = pos.ToMove;

        for (i32 i = 0; i < size; i++) {
            Move m = list[i].move;
            const auto [moveFrom, moveTo] = m.Unpack();
            const auto pt = bb.GetPieceAtIndex(moveFrom);

            if (m == ttMove) {
                list[i].score = INT32_MAX - 100000;
            }
            else if (bb.GetPieceAtIndex(moveTo) != Piece::NONE && !m.IsCastle()) {
                const auto capturedPiece = bb.GetPieceAtIndex(moveTo);
                auto& hist = history.CaptureHistory[pc][pt][moveTo][capturedPiece];
                list[i].score = (MVVMult * GetPieceValue(capturedPiece)) + hist;
            }
            else {
                i32 contIdx = MakePiece(pc, pt);

                list[i].score =  2 * history.MainHistory[pc][m.GetMoveMask()];
                list[i].score += 2 * (*(ss - 1)->ContinuationHistory)[contIdx][moveTo];
                list[i].score +=     (*(ss - 2)->ContinuationHistory)[contIdx][moveTo];
                list[i].score +=     (*(ss - 4)->ContinuationHistory)[contIdx][moveTo];
                list[i].score +=     (*(ss - 6)->ContinuationHistory)[contIdx][moveTo];

                if (pos.GivesCheck(pt, moveTo)) {
                    list[i].score += CheckBonus;
                }
            }

            if (pt == HORSIE) {
                list[i].score += 200;
            }
        }
    }

    void SearchThread::AssignScores(Position& pos, SearchStackEntry* ss, HistoryTable& history, ScoredMove* list, i32 size, Move ttMove) const {
        Bitboard& bb = pos.bb;
        const auto pc = pos.ToMove;

        const auto pawnThreats = pos.ThreatsBy<PAWN>(Not(pc));
        const auto minorThreats = pos.ThreatsBy<HORSIE>(Not(pc)) | pos.ThreatsBy<BISHOP>(Not(pc)) | pawnThreats;
        const auto rookThreats = pos.ThreatsBy<ROOK>(Not(pc)) | minorThreats;

        for (i32 i = 0; i < size; i++) {
            Move m = list[i].move;
            const auto [moveFrom, moveTo] = m.Unpack();
            const auto pt = bb.GetPieceAtIndex(moveFrom);

            if (m == ttMove) {
                list[i].score = INT32_MAX - 100000;
            }
            else if (m == ss->KillerMove) {
                list[i].score = INT32_MAX - 1000000;
            }
            else if (bb.GetPieceAtIndex(moveTo) != Piece::NONE && !m.IsCastle()) {
                const auto capturedPiece = bb.GetPieceAtIndex(moveTo);
                auto& hist = history.CaptureHistory[pc][pt][moveTo][capturedPiece];
                list[i].score = (MVVMult * GetPieceValue(capturedPiece)) + hist;
            }
            else {
                i32 contIdx = MakePiece(pc, pt);

                list[i].score =  2 * history.MainHistory[pc][m.GetMoveMask()];
                list[i].score += 2 * (*(ss - 1)->ContinuationHistory)[contIdx][moveTo];
                list[i].score +=     (*(ss - 2)->ContinuationHistory)[contIdx][moveTo];
                list[i].score +=     (*(ss - 4)->ContinuationHistory)[contIdx][moveTo];
                list[i].score +=     (*(ss - 6)->ContinuationHistory)[contIdx][moveTo];

                if (ss->Ply < LowPlyCount) {
                    list[i].score += ((2 * LowPlyCount + 1) * history.PlyHistory[ss->Ply][m.GetMoveMask()]) / (2 * ss->Ply + 1);
                }

                if (pos.GivesCheck(pt, moveTo)) {
                    list[i].score += CheckBonus;
                }

                i32 threat = 0;
                const auto fromBB = SquareBB(moveFrom);
                const auto   toBB = SquareBB(moveTo);
                if (pt == QUEEN) {
                    threat += ((fromBB & rookThreats) ? 12288 : 0);
                    threat -= ((  toBB & rookThreats) ? 11264 : 0);
                }
                else if (pt == ROOK) {
                    threat += ((fromBB & minorThreats) ? 10240 : 0);
                    threat -= ((  toBB & minorThreats) ? 9216 : 0);
                }
                else if (pt == BISHOP || pt == HORSIE) {
                    threat += ((fromBB & pawnThreats) ? 8192 : 0);
                    threat -= ((  toBB & pawnThreats) ? 7168 : 0);
                }

                list[i].score += threat;
            }

            if (pt == HORSIE) {
                list[i].score += 200;
            }
        }
    }

    std::string SearchThread::Debug_GetMovesPlayed(SearchStackEntry* ss) const {
        std::string moves = "";

        while (ss->Ply >= 0) {
            moves = ss->CurrentMove.SmithNotation(false) + " " + moves;
            ss--;
        }

        return moves;
    }
}
