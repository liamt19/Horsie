#pragma once

#ifndef UTIL_H
#define UTIL_H

#include <cstdint>
#include <string>

#include "types.h"
#include "enums.h"

using ulong = uint64_t;


namespace Horsie {

    constexpr int NormalListCapacity = 128;
    constexpr const int MoveListSize = 256;

    constexpr const int MaxDepth = 64;
    constexpr const int MaxPly = 256;

    constexpr std::string_view PieceToChar("pnbrqk ");


    constexpr short ScoreNone = 32760;
    constexpr int ScoreInfinite = 31200;
    constexpr int ScoreMate = 30000;
    constexpr int ScoreDraw = 0;
    constexpr int ScoreTTWin = ScoreMate - 512;
    constexpr int ScoreTTLoss = -ScoreTTWin;
    constexpr int ScoreMateMax = ScoreMate - 256;
    constexpr int ScoreMatedMax = -ScoreMateMax;
    constexpr int ScoreAssuredWin = 20000;
    constexpr int ScoreWin = 10000;

    constexpr int AlphaStart = -ScoreMate;
    constexpr int BetaStart = ScoreMate;



    inline int FenToPiece(char fenChar)
    {
        fenChar = char(tolower(fenChar));
        switch (fenChar)
        {
        case 'p': return PAWN;
        case 'n': return HORSIE;
        case 'b': return BISHOP;
        case 'r': return ROOK;
        case 'q': return QUEEN;
        case 'k': return KING;
        default: return NONE;
        };
    }

    inline char PieceToFEN(int pt)
    {
        switch (pt)
        {
        case PAWN  : return 'p';
        case HORSIE: return 'n';
        case BISHOP: return 'b';
        case ROOK  : return 'r';
        case QUEEN : return 'q';
        case KING: return 'k';
        default: return ' ';
        };
    }

    constexpr int GetPieceValue(int pt) {
        switch (pt) {
        case PAWN: return 199;
        case HORSIE: return 920;
        case BISHOP: return 1058;
        case ROOK: return 1553;
        case QUEEN: return 3127;
        default: return 0;
        }
    }

    constexpr int GetSEEValue(int pt) {
        switch (pt) {
        case PAWN: return 112;
        case HORSIE: return 794;
        case BISHOP: return 868;
        case ROOK: return 1324;
        case QUEEN: return 2107;
        default: return 0;
        }
    }

    constexpr int MakeMateScore(int ply) {
        return -ScoreMate + ply;
    }

    constexpr short MakeTTScore(short score, int ply)
    {
        if (score == ScoreNone)
            return score;

        if (score >= ScoreTTWin)
            return (short)(score + ply);

        if (score <= ScoreTTLoss)
            return (short)(score - ply);

        return score;
    }

    constexpr short MakeNormalScore(short ttScore, int ply)
    {
        if (ttScore == ScoreNone)
            return ttScore;

        if (ttScore >= ScoreTTWin)
            return (short)(ttScore - ply);

        if (ttScore <= ScoreTTLoss)
            return (short)(ttScore + ply);

        return ttScore;
    }



    constexpr bool IsScoreMate(int score)
    {
#if defined(_MSC_VER)
        auto v = (score < 0 ? -score : score) - ScoreMate;
        return (v < 0 ? -v : v) < MaxDepth;
#else
        return std::abs(std::abs(score) - ScoreMate) < MaxDepth;
#endif
    }

    inline std::string FormatMoveScore(int score)
    {
        if (IsScoreMate(score))
        {
            //  "mateIn" is returned in plies, but we want it in actual moves
            if (score > 0)
            {
                return "mate " + std::to_string((ScoreMate - score + 1) / 2);
            }
            else
            {
                return "mate " + std::to_string((-ScoreMate - score) / 2);
            }
        }
        else
        {
            const double NormalizeEvalFactor = 2.4;
            return "cp " + std::to_string((int)(score / NormalizeEvalFactor));
        }
    }



    const std::string EtherealFENs_D5[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1;4865609",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1;193690690",
        "4k3/8/8/8/8/8/8/4K2R w K - 0 1;133987",
        "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1;145232",
        "4k2r/8/8/8/8/8/8/4K3 w k - 0 1;47635",
        "r3k3/8/8/8/8/8/8/4K3 w q - 0 1;52710",
        "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1;532933",
        "r3k2r/8/8/8/8/8/8/4K3 w kq - 0 1;118882",
        "8/8/8/8/8/8/6k1/4K2R w K - 0 1;37735",
        "8/8/8/8/8/8/1k6/R3K3 w Q - 0 1;80619",
        "4k2r/6K1/8/8/8/8/8/8 w k - 0 1;10485",
        "r3k3/1K6/8/8/8/8/8/8 w q - 0 1;20780",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1;7594526",
        "r3k2r/8/8/8/8/8/8/1R2K2R w Kkq - 0 1;8153719",
        "r3k2r/8/8/8/8/8/8/2R1K2R w Kkq - 0 1;7736373",
        "r3k2r/8/8/8/8/8/8/R3K1R1 w Qkq - 0 1;7878456",
        "1r2k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1;8198901",
        "2r1k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1;7710115",
        "r3k1r1/8/8/8/8/8/8/R3K2R w KQq - 0 1;7848606",
        "4k3/8/8/8/8/8/8/4K2R b K - 0 1;47635",
        "4k3/8/8/8/8/8/8/R3K3 b Q - 0 1;52710",
        "4k2r/8/8/8/8/8/8/4K3 b k - 0 1;133987",
        "r3k3/8/8/8/8/8/8/4K3 b q - 0 1;145232",
        "4k3/8/8/8/8/8/8/R3K2R b KQ - 0 1;118882",
        "r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1;532933",
        "8/8/8/8/8/8/6k1/4K2R b K - 0 1;10485",
        "8/8/8/8/8/8/1k6/R3K3 b Q - 0 1;20780",
        "4k2r/6K1/8/8/8/8/8/8 b k - 0 1;37735",
        "r3k3/1K6/8/8/8/8/8/8 b q - 0 1;80619",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1;7594526",
        "r3k2r/8/8/8/8/8/8/1R2K2R b Kkq - 0 1;8198901",
        "r3k2r/8/8/8/8/8/8/2R1K2R b Kkq - 0 1;7710115",
        "r3k2r/8/8/8/8/8/8/R3K1R1 b Qkq - 0 1;7848606",
        "1r2k2r/8/8/8/8/8/8/R3K2R b KQk - 0 1;8153719",
        "2r1k2r/8/8/8/8/8/8/R3K2R b KQk - 0 1;7736373",
        "r3k1r1/8/8/8/8/8/8/R3K2R b KQq - 0 1;7878456",
        "8/1n4N1/2k5/8/8/5K2/1N4n1/8 w - - 0 1;570726",
        "8/1k6/8/5N2/8/4n3/8/2K5 w - - 0 1;223507",
        "8/8/4k3/3Nn3/3nN3/4K3/8/8 w - - 0 1;1198299",
        "K7/8/2n5/1n6/8/8/8/k6N w - - 0 1;38348",
        "k7/8/2N5/1N6/8/8/8/K6n w - - 0 1;92250",
        "8/1n4N1/2k5/8/8/5K2/1N4n1/8 b - - 0 1;582642",
        "8/1k6/8/5N2/8/4n3/8/2K5 b - - 0 1;288141",
        "8/8/3K4/3Nn3/3nN3/4k3/8/8 b - - 0 1;281190",
        "K7/8/2n5/1n6/8/8/8/k6N b - - 0 1;92250",
        "k7/8/2N5/1N6/8/8/8/K6n b - - 0 1;38348",
        "B6b/8/8/8/2K5/4k3/8/b6B w - - 0 1;1320507",
        "8/8/1B6/7b/7k/8/2B1b3/7K w - - 0 1;1713368",
        "k7/B7/1B6/1B6/8/8/8/K6b w - - 0 1;787524",
        "K7/b7/1b6/1b6/8/8/8/k6B w - - 0 1;310862",
        "B6b/8/8/8/2K5/5k2/8/b6B b - - 0 1;530585",
        "8/8/1B6/7b/7k/8/2B1b3/7K b - - 0 1;1591064",
        "k7/B7/1B6/1B6/8/8/8/K6b b - - 0 1;310862",
        "K7/b7/1b6/1b6/8/8/8/k6B b - - 0 1;787524",
        "7k/RR6/8/8/8/8/rr6/7K w - - 0 1;2161211",
        "R6r/8/8/2K5/5k2/8/8/r6R w - - 0 1;20506480",
        "7k/RR6/8/8/8/8/rr6/7K b - - 0 1;2161211",
        "R6r/8/8/2K5/5k2/8/8/r6R b - - 0 1;20521342",
        "6kq/8/8/8/8/8/8/7K w - - 0 1;14893",
        "6KQ/8/8/8/8/8/8/7k b - - 0 1;14893",
        "K7/8/8/3Q4/4q3/8/8/7k w - - 0 1;166741",
        "6qk/8/8/8/8/8/8/7K b - - 0 1;105749",
        "6KQ/8/8/8/8/8/8/7k b - - 0 1;14893",
        "K7/8/8/3Q4/4q3/8/8/7k b - - 0 1;166741",
        "8/8/8/8/8/K7/P7/k7 w - - 0 1;1347",
        "8/8/8/8/8/7K/7P/7k w - - 0 1;1347",
        "K7/p7/k7/8/8/8/8/8 w - - 0 1;342",
        "7K/7p/7k/8/8/8/8/8 w - - 0 1;342",
        "8/2k1p3/3pP3/3P2K1/8/8/8/8 w - - 0 1;7028",
        "8/8/8/8/8/K7/P7/k7 b - - 0 1;342",
        "8/8/8/8/8/7K/7P/7k b - - 0 1;342",
        "K7/p7/k7/8/8/8/8/8 b - - 0 1;1347",
        "7K/7p/7k/8/8/8/8/8 b - - 0 1;1347",
        "8/2k1p3/3pP3/3P2K1/8/8/8/8 b - - 0 1;5408",
        "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1;1814",
        "4k3/4p3/4K3/8/8/8/8/8 b - - 0 1;1814",
        "8/8/7k/7p/7P/7K/8/8 w - - 0 1;1969",
        "8/8/k7/p7/P7/K7/8/8 w - - 0 1;1969",
        "8/8/3k4/3p4/3P4/3K4/8/8 w - - 0 1;8296",
        "8/3k4/3p4/8/3P4/3K4/8/8 w - - 0 1;23599",
        "8/8/3k4/3p4/8/3P4/3K4/8 w - - 0 1;21637",
        "k7/8/3p4/8/3P4/8/8/7K w - - 0 1;3450",
        "8/8/7k/7p/7P/7K/8/8 b - - 0 1;1969",
        "8/8/k7/p7/P7/K7/8/8 b - - 0 1;1969",
        "8/8/3k4/3p4/3P4/3K4/8/8 b - - 0 1;8296",
        "8/3k4/3p4/8/3P4/3K4/8/8 b - - 0 1;21637",
        "8/8/3k4/3p4/8/3P4/3K4/8 b - - 0 1;23599",
        "k7/8/3p4/8/3P4/8/8/7K b - - 0 1;3309",
        "7k/3p4/8/8/3P4/8/8/K7 w - - 0 1;4661",
        "7k/8/8/3p4/8/8/3P4/K7 w - - 0 1;4786",
        "k7/8/8/7p/6P1/8/8/K7 w - - 0 1;6112",
        "k7/8/7p/8/8/6P1/8/K7 w - - 0 1;4354",
        "k7/8/8/6p1/7P/8/8/K7 w - - 0 1;6112",
        "k7/8/6p1/8/8/7P/8/K7 w - - 0 1;4354",
        "k7/8/8/3p4/4p3/8/8/7K w - - 0 1;3013",
        "k7/8/3p4/8/8/4P3/8/7K w - - 0 1;4271",
        "7k/3p4/8/8/3P4/8/8/K7 b - - 0 1;5014",
        "7k/8/8/3p4/8/8/3P4/K7 b - - 0 1;4658",
        "k7/8/8/7p/6P1/8/8/K7 b - - 0 1;6112",
        "k7/8/7p/8/8/6P1/8/K7 b - - 0 1;4354",
        "k7/8/8/6p1/7P/8/8/K7 b - - 0 1;6112",
        "k7/8/6p1/8/8/7P/8/K7 b - - 0 1;4354",
        "k7/8/8/3p4/4p3/8/8/7K b - - 0 1;4337",
        "k7/8/3p4/8/8/4P3/8/7K b - - 0 1;4271",
        "7k/8/8/p7/1P6/8/8/7K w - - 0 1;6112",
        "7k/8/p7/8/8/1P6/8/7K w - - 0 1;4354",
        "7k/8/8/1p6/P7/8/8/7K w - - 0 1;6112",
        "7k/8/1p6/8/8/P7/8/7K w - - 0 1;4354",
        "k7/7p/8/8/8/8/6P1/K7 w - - 0 1;7574",
        "k7/6p1/8/8/8/8/7P/K7 w - - 0 1;7574",
        "3k4/3pp3/8/8/8/8/3PP3/3K4 w - - 0 1;24122",
        "7k/8/8/p7/1P6/8/8/7K b - - 0 1;6112",
        "7k/8/p7/8/8/1P6/8/7K b - - 0 1;4354",
        "7k/8/8/1p6/P7/8/8/7K b - - 0 1;6112",
        "7k/8/1p6/8/8/P7/8/7K b - - 0 1;4354",
        "k7/7p/8/8/8/8/6P1/K7 b - - 0 1;7574",
        "k7/6p1/8/8/8/8/7P/K7 b - - 0 1;7574",
        "3k4/3pp3/8/8/8/8/3PP3/3K4 b - - 0 1;24122",
        "8/Pk6/8/8/8/8/6Kp/8 w - - 0 1;90606",
        "n1n5/1Pk5/8/8/8/8/5Kp1/5N1N w - - 0 1;2193768",
        "8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1;1533145",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1;3605103",
        "8/Pk6/8/8/8/8/6Kp/8 b - - 0 1;90606",
        "n1n5/1Pk5/8/8/8/8/5Kp1/5N1N b - - 0 1;2193768",
        "8/PPPk4/8/8/8/8/4Kppp/8 b - - 0 1;1533145",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1;3605103",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1;674624",
    };

}

#endif // !UTIL_H