#pragma once

#ifndef SEARCH_OPTIONS_H
#define SEARCH_OPTIONS_H

#include "tunable.h"

namespace Horsie {

    UCI_OPTION_SPECIAL(Threads, 1, 1, 512)
    UCI_OPTION_SPECIAL(Hash, 32, 1, 1048576)
    UCI_OPTION_SPECIAL(MultiPV, 1, 1, 1)
    UCI_OPTION_SPIN(UCI_Chess960, false)

    const i32 MoveOverhead = 25;

    const bool ShallowPruning = true;
    const bool UseSingularExtensions = true;
    const bool UseNMP = true;
    const bool UseRFP = true;
    const bool UseProbcut = true;

    const i32 CorrectionScale = 1024;
    const i32 CorrectionGrain = 256;
    const i32 CorrectionMax = CorrectionGrain * 64;

    UCI_OPTION(SEMinDepth, 5)
    UCI_OPTION(SENumerator, 11)
    UCI_OPTION(SEDoubleMargin, 24)
    UCI_OPTION(SETripleMargin, 97)
    UCI_OPTION(SETripleCapSub, 73)
    UCI_OPTION_CUSTOM(SEDepthAdj, -1, -3, 2)

    UCI_OPTION(NMPMinDepth, 6)
    UCI_OPTION(NMPBaseRed, 4)
    UCI_OPTION(NMPDepthDiv, 4)
    UCI_OPTION(NMPEvalDiv, 160)
    UCI_OPTION_CUSTOM(NMPEvalMin, 2, 0, 6)

    UCI_OPTION(RFPMaxDepth, 6)
    UCI_OPTION(RFPMargin, 44)

    //UCI_OPTION_CUSTOM(ProbcutMinDepth, 1, 1, 5)
    UCI_OPTION(ProbcutBeta, 259)
    UCI_OPTION(ProbcutBetaImp, 89)

    UCI_OPTION(NMFutileBase, 465)
    UCI_OPTION(NMFutilePVCoeff, 1058)
    UCI_OPTION(NMFutileImpCoeff, 1014)
    UCI_OPTION(NMFutileHistCoeff, 1080)
    UCI_OPTION(NMFutMarginB, 186)
    UCI_OPTION(NMFutMarginM, 85)
    UCI_OPTION(NMFutMarginDiv, 138)
    UCI_OPTION(ShallowSEEMargin, 214)
    UCI_OPTION(ShallowMaxDepth, 9)

    UCI_OPTION(LMRQuietDiv, 12655)
    UCI_OPTION(LMRCaptureDiv, 9328)
    UCI_OPTION(LMRExtMargin, 135)

    UCI_OPTION(QSFutileMargin, 183)
    UCI_OPTION(QSSeeMargin, 78)

    UCI_OPTION(CheckBonus, 9315)
    UCI_OPTION(MVVMult, 11)

    UCI_OPTION_CUSTOM(IIRMinDepth, 3, 2, 6)
    UCI_OPTION(AspWindow, 12)

    UCI_OPTION(StatBonusMult, 178)
    UCI_OPTION(StatBonusSub, 79)
    UCI_OPTION(StatBonusMax, 1744)

    UCI_OPTION(StatMalusMult, 665)
    UCI_OPTION(StatMalusSub, 103)
    UCI_OPTION(StatMalusMax, 1471)

    UCI_OPTION(SEEValuePawn, 105)
    UCI_OPTION(SEEValueHorsie, 900)
    UCI_OPTION(SEEValueBishop, 1054)
    UCI_OPTION(SEEValueRook, 1332);
    UCI_OPTION(SEEValueQueen, 2300);

    UCI_OPTION(ValuePawn, 171)
    UCI_OPTION(ValueHorsie, 794)
    UCI_OPTION(ValueBishop, 943)
    UCI_OPTION(ValueRook, 1620)
    UCI_OPTION(ValueQueen, 2994)

}


#endif