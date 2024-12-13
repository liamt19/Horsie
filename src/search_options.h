#pragma once

#ifndef SEARCH_OPTIONS_H
#define SEARCH_OPTIONS_H

#include "tunable.h"

namespace Horsie {

    TUNABLE_PARAM_SPECIAL(Threads, 1, 1, 512)
    TUNABLE_PARAM_SPECIAL(MultiPV, 1, 1, 256)
    TUNABLE_PARAM_SPECIAL(Hash, 32, 1, 1048576)

    static i32 MoveOverhead = 25;
    static bool UCI_Chess960 = false;

    const bool ShallowPruning = true;
    const bool UseSingularExtensions = true;
    const bool UseNMP = true;
    const bool UseRFP = true;
    const bool UseProbCut = true;

    const i32 CorrectionScale = 1024;
    const i32 CorrectionGrain = 256;
    const i32 CorrectionMax = CorrectionGrain * 64;

    TUNABLE_PARAM(SEMinDepth, 5)
    TUNABLE_PARAM(SENumerator, 11)
    TUNABLE_PARAM(SEDoubleMargin, 24)
    TUNABLE_PARAM(SETripleMargin, 96)
    TUNABLE_PARAM(SETripleCapSub, 75)
    TUNABLE_PARAM_CUSTOM(SEDepthAdj, -1, -3, 2)

    TUNABLE_PARAM(NMPMinDepth, 6)
    TUNABLE_PARAM(NMPBaseRed, 4)
    TUNABLE_PARAM(NMPDepthDiv, 4)
    TUNABLE_PARAM(NMPEvalDiv, 181)
    TUNABLE_PARAM_CUSTOM(NMPEvalMin, 2, 0, 6)

    TUNABLE_PARAM(RFPMaxDepth, 6)
    TUNABLE_PARAM(RFPMargin, 47)

    TUNABLE_PARAM_CUSTOM(ProbCutMinDepth, 2, 1, 5)
    TUNABLE_PARAM(ProbCutBeta, 256)
    TUNABLE_PARAM(ProbcutBetaImp, 101)

    TUNABLE_PARAM(ShallowSEEMargin, 216)
    TUNABLE_PARAM(ShallowMaxDepth, 9)

    TUNABLE_PARAM(LMRQuietDiv, 12288)
    TUNABLE_PARAM(LMRCaptureDiv, 10288)
    TUNABLE_PARAM(LMRExtMargin, 128)

    TUNABLE_PARAM(QSFutileMargin, 183)
    TUNABLE_PARAM(QSSeeMargin, 81)

    TUNABLE_PARAM(CheckBonus, 9546)
    TUNABLE_PARAM(MVVMult, 12)

    TUNABLE_PARAM_CUSTOM(IIRMinDepth, 3, 2, 6)
    TUNABLE_PARAM(AspWindow, 12)

    TUNABLE_PARAM(StatBonusMult, 184)
    TUNABLE_PARAM(StatBonusSub, 80)
    TUNABLE_PARAM(StatBonusMax, 1667)

    TUNABLE_PARAM(StatMalusMult, 611)
    TUNABLE_PARAM(StatMalusSub, 111)
    TUNABLE_PARAM(StatMalusMax, 1663)

    TUNABLE_PARAM(SEEValue_Pawn, 105)
    TUNABLE_PARAM(SEEValue_Horsie, 900)
    TUNABLE_PARAM(SEEValue_Bishop, 1054)
    TUNABLE_PARAM(SEEValue_Rook, 1332);
    TUNABLE_PARAM(SEEValue_Queen, 2300);

    TUNABLE_PARAM(ValuePawn, 171)
    TUNABLE_PARAM(ValueHorsie, 794)
    TUNABLE_PARAM(ValueBishop, 943)
    TUNABLE_PARAM(ValueRook, 1620)
    TUNABLE_PARAM(ValueQueen, 2994)

}


#endif