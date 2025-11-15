#pragma once

#include "defs.h"
#include "tunable.h"

namespace Horsie {

    UCI_OPTION_SPECIAL(Threads, 1, 1, 2048)
    UCI_OPTION_SPECIAL(Hash, 32, 1, 1048576)
    UCI_OPTION_SPECIAL(MultiPV, 1, 1, 256)
    UCI_OPTION_SPECIAL(MoveOverhead, 25, 1, 5000)
    UCI_OPTION_SPIN(UCI_Chess960, false)
    UCI_OPTION_SPIN(UCI_ShowWDL, true)

    const bool ShallowPruning = true;
    const bool UseSingularExtensions = true;
    const bool UseNMP = true;
    const bool UseRazoring = true;
    const bool UseRFP = true;
    const bool UseProbcut = true;

    const i32 CorrectionScale = 1024;
    const i32 CorrectionGrain = 256;
    const i32 CorrectionMax = CorrectionGrain * 64;

    UCI_OPTION(QuietOrderMin, 101)
    UCI_OPTION(QuietOrderMax, 192)
    UCI_OPTION(QuietOrderMult, 153)

    UCI_OPTION(SEMinDepth, 5)
    UCI_OPTION(SENumerator, 11)
    UCI_OPTION(SEDoubleMargin, 21)
    UCI_OPTION(SETripleMargin, 101)
    UCI_OPTION(SETripleCapSub, 73)
    UCI_OPTION_CUSTOM(SEDepthAdj, -1, -3, 2)

    UCI_OPTION(NMPMinDepth, 6)
    UCI_OPTION(NMPBaseRed, 4)
    UCI_OPTION(NMPDepthDiv, 4)
    UCI_OPTION(NMPEvalDiv, 151)
    UCI_OPTION_CUSTOM(NMPEvalMin, 2, 0, 6)

    UCI_OPTION(RazoringMaxDepth, 4)
    UCI_OPTION(RazoringMult, 292)

    UCI_OPTION(RFPMaxDepth, 6)
    UCI_OPTION(RFPMargin, 51)

    //UCI_OPTION_CUSTOM(ProbcutMinDepth, 1, 1, 5)
    UCI_OPTION(ProbcutBeta, 252)
    UCI_OPTION(ProbcutBetaImp, 97)

    UCI_OPTION(NMFutileBase, 58)
    UCI_OPTION(NMFutilePVCoeff, 136)
    UCI_OPTION(NMFutileImpCoeff, 132)
    UCI_OPTION(NMFutileHistCoeff, 128)
    UCI_OPTION(NMFutMarginB, 163)
    UCI_OPTION(NMFutMarginM, 81)
    UCI_OPTION(NMFutMarginDiv, 129)
    UCI_OPTION(ShallowSEEMargin, 81)
    UCI_OPTION(ShallowMaxDepth, 9)

    UCI_OPTION(LMRNotImpCoeff, 119)
    UCI_OPTION(LMRCutNodeCoeff, 280)
    UCI_OPTION(LMRTTPVCoeff, 124)
    UCI_OPTION(LMRPVCoeff, 133)
    UCI_OPTION(LMRKillerCoeff, 129)

    UCI_OPTION(LMRHist, 189)
    UCI_OPTION(LMRHistSS1, 268)
    UCI_OPTION(LMRHistSS2, 128)
    UCI_OPTION(LMRHistSS4, 127)

    UCI_OPTION(PawnCorrCoeff, 129)
    UCI_OPTION(NonPawnCorrCoeff, 70)
    UCI_OPTION(CorrDivisor, 3136)

    UCI_OPTION(NMOrderingMH, 419)
    UCI_OPTION(NMOrderingSS1, 491)
    UCI_OPTION(NMOrderingSS2, 258)
    UCI_OPTION(NMOrderingSS4, 255)
    UCI_OPTION(NMOrderingSS6, 246)

    UCI_OPTION(QSOrderingMH, 498)
    UCI_OPTION(QSOrderingSS1, 500)
    UCI_OPTION(QSOrderingSS2, 258)
    UCI_OPTION(QSOrderingSS4, 247)
    UCI_OPTION(QSOrderingSS6, 261)

    UCI_OPTION(OrderingEnPriseMult, 520)

    UCI_OPTION(LMRQuietDiv, 15234)
    UCI_OPTION(LMRCaptureDiv, 9229)

    UCI_OPTION(DeeperMargin, 45)

    UCI_OPTION(QSFutileMargin, 204)
    UCI_OPTION(QSSeeMargin, 75)

    UCI_OPTION(CheckBonus, 9816)
    UCI_OPTION(MVVMult, 363)

    UCI_OPTION_CUSTOM(IIRMinDepth, 3, 2, 6)
    UCI_OPTION(AspWindow, 12)

    UCI_OPTION(StatBonusMult, 192)
    UCI_OPTION(StatBonusSub, 91)
    UCI_OPTION(StatBonusMax, 1735)

    UCI_OPTION(StatPenaltyMult, 655)
    UCI_OPTION(StatPenaltySub, 106)
    UCI_OPTION(StatPenaltyMax, 1267)

    UCI_OPTION(LMRBonusMult, 182)
    UCI_OPTION(LMRBonusSub, 83)
    UCI_OPTION(LMRBonusMax, 1703)

    UCI_OPTION(LMRPenaltyMult, 173)
    UCI_OPTION(LMRPenaltySub, 82)
    UCI_OPTION(LMRPenaltyMax, 1569)

    CONST_OPTION(SEEValuePawn, 105)
    CONST_OPTION(SEEValueHorsie, 300)
    CONST_OPTION(SEEValueBishop, 315)
    CONST_OPTION(SEEValueRook, 535)
    CONST_OPTION(SEEValueQueen, 970)

    UCI_OPTION(ValuePawn, 161)
    UCI_OPTION(ValueHorsie, 737)
    UCI_OPTION(ValueBishop, 901)
    UCI_OPTION(ValueRook, 1518)
    UCI_OPTION(ValueQueen, 3200)

}
