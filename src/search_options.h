#pragma once

#ifndef SEARCH_OPTIONS_H
#define SEARCH_OPTIONS_H

namespace Horsie {

    static int Threads = 1;

    static int MultiPV = 1;

    static int Hash = 32;

    static bool UCI_Chess960 = false;

    const bool UseSingularExtensions = false;
    static int SingularExtensionsMinDepth = 7;
    static int SingularExtensionsNumerator = 9;
    static int SingularExtensionsBeta = 22;
    static int SingularExtensionsDepthAugment = -1;

    const bool UseNullMovePruning = false;
    static int NMPMinDepth = 6;
    static int NMPReductionBase = 5;
    static int NMPReductionDivisor = 5;

    const bool UseReverseFutilityPruning = false;
    static int ReverseFutilityPruningMaxDepth = 7;
    static int ReverseFutilityPruningPerDepth = 47;

    const bool UseProbCut = false;
    static int ProbCutBeta = 191;
    static int ProbCutBetaImproving = 100;
    static int ProbCutMinDepth = 2;

    static int LMRExtensionThreshold = 131;
    static int LMRExchangeBase = 216;

    static int HistoryReductionMultiplier = 3;

    static int FutilityExchangeBase = 181;

    static int ExtraCutNodeReductionMinDepth = 5;

    static int AspirationWindowMargin = 11;

    static int HistoryCaptureBonusMargin = 158;
    static int OrderingGivesCheckBonus = 10345;
    static int OrderingVictimValueMultiplier = 14;
    static int OrderingHistoryDivisor = 11;

    static int StatBonusMult = 170;
    static int StatBonusSub = 95;
    static int StatBonusMax = 1822;

    static int StatMalusMult = 466;
    static int StatMalusSub = 97;
    static int StatMalusMax = 1787;

    static int SEEValue_Pawn = 112;
    static int SEEValue_Knight = 794;
    static int SEEValue_Bishop = 868;
    static int SEEValue_Rook = 1324;
    static int SEEValue_Queen = 2107;

    static int ValuePawn = 199;
    static int ValueKnight = 920;
    static int ValueBishop = 1058;
    static int ValueRook = 1553;
    static int ValueQueen = 3127;


    inline int GetReverseFutilityMargin(int depth, bool improving)
    {
        return (depth - (improving ? 1 : 0)) * ReverseFutilityPruningPerDepth;
    }

}


#endif