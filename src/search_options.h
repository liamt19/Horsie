#pragma once

#ifndef SEARCH_OPTIONS_H
#define SEARCH_OPTIONS_H

namespace Horsie {

    static i32 Threads = 1;

    static i32 MultiPV = 1;

    static i32 Hash = 32;

    static bool UCI_Chess960 = false;

    const bool ShallowPruning = true;
    const bool UseSingularExtensions = true;
    const bool UseNMP = true;
    const bool UseRFP = true;
    const bool UseProbCut = true;

    const i32 CorrectionScale = 1024;
    const i32 CorrectionGrain = 256;
    const i32 CorrectionMax = CorrectionGrain * 64;

    static i32 SEMinDepth = 5;
    static i32 SENumerator = 11;
    static i32 SEDoubleMargin = 24;
    static i32 SETripleMargin = 96;
    static i32 SETripleCapSub = 75;
    static i32 SEDepthAdj = -1;

    static i32 NMPMinDepth = 6;
    static i32 NMPBaseRed = 4;
    static i32 NMPDepthDiv = 4;
    static i32 NMPEvalDiv = 181;
    static i32 NMPEvalMin = 2;

    static i32 RFPMaxDepth = 6;
    static i32 RFPMargin = 47;

    static i32 ProbCutMinDepth = 2;
    static i32 ProbCutBeta = 256;
    static i32 ProbcutBetaImp = 101;

    static i32 ShallowSEEMargin = 216;
    static i32 ShallowMaxDepth = 9;

    static i32 LMRQuietDiv = 12288;
    static i32 LMRCaptureDiv = 10288;
    static i32 LMRExtMargin = 48;

    static i32 QSFutileMargin = 183;
    static i32 QSSeeMargin = 81;

    static i32 OrderingGivesCheckBonus = 9546;
    static i32 OrderingVictimValueMultiplier = 14;

    static i32 IIRMinDepth = 3;
    static i32 AspWindow = 12;

    static i32 StatBonusMult = 184;
    static i32 StatBonusSub = 80;
    static i32 StatBonusMax = 1667;

    static i32 StatMalusMult = 611;
    static i32 StatMalusSub = 111;
    static i32 StatMalusMax = 1663;

    static i32 SEEValue_Pawn = 105;
    static i32 SEEValue_Horsie = 900;
    static i32 SEEValue_Bishop = 1054;
    static i32 SEEValue_Rook = 1332;
    static i32 SEEValue_Queen = 2300;

    static i32 ValuePawn = 171;
    static i32 ValueHorsie = 794;
    static i32 ValueBishop = 943;
    static i32 ValueRook = 1620;
    static i32 ValueQueen = 2994;

}


#endif