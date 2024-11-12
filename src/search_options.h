#pragma once

#ifndef SEARCH_OPTIONS_H
#define SEARCH_OPTIONS_H

namespace Horsie {

    static int Threads = 1;

    static int MultiPV = 1;

    static int Hash = 32;

    static bool UCI_Chess960 = false;

    const bool ShallowPruning = true;
    const bool UseSingularExtensions = true;
    const bool UseNMP = true;
    const bool UseRFP = true;
    const bool UseProbCut = true;

    const int CorrectionScale = 1024;
    const int CorrectionGrain = 256;
    const int CorrectionMax = CorrectionGrain * 64;

    static int SEMinDepth = 5;
    static int SENumerator = 11;
    static int SEDoubleMargin = 24;
    static int SETripleMargin = 96;
    static int SETripleCapSub = 75;
    static int SEDepthAdj = -1;

    static int NMPMinDepth = 6;
    static int NMPBaseRed = 4;
    static int NMPDepthDiv = 4;
    static int NMPEvalDiv = 181;
    static int NMPEvalMin = 2;

    static int RFPMaxDepth = 6;
    static int RFPMargin = 47;

    static int ProbCutBeta = 256;
    static int ProbcutBetaImp = 101;
    static int ProbCutMinDepth = 2;

    static int ShallowSEEMargin = 216;
    static int ShallowMaxDepth = 9;

    static int LMRQuietDiv = 12288;
    static int LMRCaptureDiv = 10288;
    static int LMRExtMargin = 128;

    static int QSFutileMargin = 183;
    static int QSSeeMargin = 81;

    static int OrderingGivesCheckBonus = 9546;
    static int OrderingVictimValueMultiplier = 14;

    static int IIRMinDepth = 4;
    static int AspWindow = 11;
    static int HistoryCaptureBonusMargin = 158;

    static int StatBonusMult = 184;
    static int StatBonusSub = 80;
    static int StatBonusMax = 1667;

    static int StatMalusMult = 611;
    static int StatMalusSub = 111;
    static int StatMalusMax = 1663;

    static int SEEValue_Pawn = 105;
    static int SEEValue_Knight = 900;
    static int SEEValue_Bishop = 1054;
    static int SEEValue_Rook = 1332;
    static int SEEValue_Queen = 2300;

    static int ValuePawn = 171;
    static int ValueKnight = 794;
    static int ValueBishop = 943;
    static int ValueRook = 1620;
    static int ValueQueen = 2994;

}


#endif