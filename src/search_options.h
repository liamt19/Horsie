#pragma once

#ifndef SEARCH_OPTIONS_H
#define SEARCH_OPTIONS_H

#include "defs.h"
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
    UCI_OPTION(SETripleMargin, 96)
    UCI_OPTION(SETripleCapSub, 75)
    UCI_OPTION_CUSTOM(SEDepthAdj, -1, -3, 2)

    UCI_OPTION(NMPMinDepth, 6)
    UCI_OPTION(NMPBaseRed, 4)
    UCI_OPTION(NMPDepthDiv, 4)
    UCI_OPTION(NMPEvalDiv, 181)
    UCI_OPTION_CUSTOM(NMPEvalMin, 2, 0, 6)

    UCI_OPTION(RFPMaxDepth, 6)
    UCI_OPTION(RFPMargin, 47)

    //UCI_OPTION_CUSTOM(ProbcutMinDepth, 1, 1, 5)
    UCI_OPTION(ProbcutBeta, 256)
    UCI_OPTION(ProbcutBetaImp, 101)

    UCI_OPTION(NMFutileBase, 512)
    UCI_OPTION(NMFutilePVCoeff, 1024)
    UCI_OPTION(NMFutileImpCoeff, 1024)
    UCI_OPTION(NMFutileHistCoeff, 1024)
    UCI_OPTION(NMFutMarginB, 200)
    UCI_OPTION(NMFutMarginM, 80)
    UCI_OPTION(NMFutMarginDiv, 150)
    UCI_OPTION(ShallowSEEMargin, 216)
    UCI_OPTION(ShallowMaxDepth, 9)

    UCI_OPTION(LMRQuietDiv, 12288)
    UCI_OPTION(LMRCaptureDiv, 10288)
    UCI_OPTION(LMRExtMargin, 128)

    UCI_OPTION(QSFutileMargin, 183)
    UCI_OPTION(QSSeeMargin, 81)

    UCI_OPTION(CheckBonus, 9546)
    UCI_OPTION(MVVMult, 12)

    UCI_OPTION_CUSTOM(IIRMinDepth, 3, 2, 6)
    UCI_OPTION(AspWindow, 12)

    UCI_OPTION(StatBonusMult, 184)
    UCI_OPTION(StatBonusSub, 80)
    UCI_OPTION(StatBonusMax, 1667)

    UCI_OPTION(StatMalusMult, 611)
    UCI_OPTION(StatMalusSub, 111)
    UCI_OPTION(StatMalusMax, 1663)

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


#endif // !SEARCH_OPTIONS_H