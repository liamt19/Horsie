#pragma once

#ifndef UTIL_H
#define UTIL_H

#include <cstdint>
#include <string>
#include <sstream>

#include <iomanip>
#include <locale>

#include "types.h"
#include "enums.h"
#include "search_options.h"

using u64 = uint64_t;


namespace Horsie {

    constexpr i32 NormalListCapacity = 128;
    constexpr const i32 MoveListSize = 256;

    constexpr const i32 MaxDepth = 64;
    constexpr const i32 MaxPly = 256;

    constexpr std::string_view PieceToChar("pnbrqk ");


    constexpr i16 ScoreNone = 32760;
    constexpr i32 ScoreInfinite = 31200;
    constexpr i32 ScoreMate = 30000;
    constexpr i32 ScoreDraw = 0;
    constexpr i32 ScoreTTWin = ScoreMate - 512;
    constexpr i32 ScoreTTLoss = -ScoreTTWin;
    constexpr i32 ScoreMateMax = ScoreMate - 256;
    constexpr i32 ScoreMatedMax = -ScoreMateMax;
    constexpr i32 ScoreAssuredWin = 20000;
    constexpr i32 ScoreWin = 10000;

    constexpr i32 AlphaStart = -ScoreMate;
    constexpr i32 BetaStart = ScoreMate;



    inline i32 FenToPiece(char fenChar)
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

    inline char PieceToFEN(i32 pt)
    {
        switch (pt)
        {
        case PAWN  : return 'p';
        case HORSIE: return 'n';
        case BISHOP: return 'b';
        case ROOK  : return 'r';
        case QUEEN : return 'q';
        case KING  : return 'k';
        default:     return ' ';
        };
    }

    constexpr i32 GetPieceValue(i32 pt) {
        switch (pt) {
        case PAWN  : return ValuePawn;
        case HORSIE: return ValueHorsie;
        case BISHOP: return ValueBishop;
        case ROOK  : return ValueRook;
        case QUEEN : return ValueQueen;
        default:     return 0;
        }
    }

    constexpr i32 GetSEEValue(i32 pt) {
        switch (pt) {
        case PAWN  : return SEEValue_Pawn;
        case HORSIE: return SEEValue_Horsie;
        case BISHOP: return SEEValue_Bishop;
        case ROOK  : return SEEValue_Rook;
        case QUEEN : return SEEValue_Queen;
        default:     return 0;
        }
    }

    constexpr i32 MakeDrawScore(u64 nodes) {
        return -1 + (i32)(nodes & 2);
    }

    constexpr i32 MakeMateScore(i32 ply) {
        return -ScoreMate + ply;
    }

    constexpr i16 MakeTTScore(i16 score, i32 ply)
    {
        if (score == ScoreNone)
            return score;

        if (score >= ScoreTTWin)
            return (i16)(score + ply);

        if (score <= ScoreTTLoss)
            return (i16)(score - ply);

        return score;
    }

    constexpr i16 MakeNormalScore(i16 ttScore, i32 ply)
    {
        if (ttScore == ScoreNone)
            return ttScore;

        if (ttScore >= ScoreTTWin)
            return (i16)(ttScore - ply);

        if (ttScore <= ScoreTTLoss)
            return (i16)(ttScore + ply);

        return ttScore;
    }



    constexpr bool IsScoreMate(i32 score)
    {
#if defined(_MSC_VER)
        auto v = (score < 0 ? -score : score) - ScoreMate;
        return (v < 0 ? -v : v) < MaxDepth;
#else
        return std::abs(std::abs(score) - ScoreMate) < MaxDepth;
#endif
    }

    inline std::string FormatMoveScore(i32 score)
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
            const double NormalizeEvalFactor = 252;
            return "cp " + std::to_string((i32)((score * 100) / NormalizeEvalFactor));
        }
    }

    template<class T>
    inline std::string FormatWithCommas(T value)
    {
        std::stringstream ss;
        ss.imbue(std::locale(""));
        ss << std::fixed << value;
        return ss.str();
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

    const std::string BenchFENs[] = {
        "r3k2r/2pb1ppp/2pp1q2/p7/1nP1B3/1P2P3/P2N1PPP/R2QK2R w KQkq a6 0 14",
        "4rrk1/2p1b1p1/p1p3q1/4p3/2P2n1p/1P1NR2P/PB3PP1/3R1QK1 b - - 2 24",
        "r3qbrk/6p1/2b2pPp/p3pP1Q/PpPpP2P/3P1B2/2PB3K/R5R1 w - - 16 42",
        "6k1/1R3p2/6p1/2Bp3p/3P2q1/P7/1P2rQ1K/5R2 b - - 4 44",
        "8/8/1p2k1p1/3p3p/1p1P1P1P/1P2PK2/8/8 w - - 3 54",
        "7r/2p3k1/1p1p1qp1/1P1Bp3/p1P2r1P/P7/4R3/Q4RK1 w - - 0 36",
        "r1bq1rk1/pp2b1pp/n1pp1n2/3P1p2/2P1p3/2N1P2N/PP2BPPP/R1BQ1RK1 b - - 2 10",
        "3r3k/2r4p/1p1b3q/p4P2/P2Pp3/1B2P3/3BQ1RP/6K1 w - - 3 87",
        "2r4r/1p4k1/1Pnp4/3Qb1pq/8/4BpPp/5P2/2RR1BK1 w - - 0 42",
        "4q1bk/6b1/7p/p1p4p/PNPpP2P/KN4P1/3Q4/4R3 b - - 0 37",
        "2q3r1/1r2pk2/pp3pp1/2pP3p/P1Pb1BbP/1P4Q1/R3NPP1/4R1K1 w - - 2 34",
        "1r2r2k/1b4q1/pp5p/2pPp1p1/P3Pn2/1P1B1Q1P/2R3P1/4BR1K b - - 1 37",
        "r3kbbr/pp1n1p1P/3ppnp1/q5N1/1P1pP3/P1N1B3/2P1QP2/R3KB1R b KQkq b3 0 17",
        "8/6pk/2b1Rp2/3r4/1R1B2PP/P5K1/8/2r5 b - - 16 42",
        "1r4k1/4ppb1/2n1b1qp/pB4p1/1n1BP1P1/7P/2PNQPK1/3RN3 w - - 8 29",
        "8/p2B4/PkP5/4p1pK/4Pb1p/5P2/8/8 w - - 29 68",
        "3r4/ppq1ppkp/4bnp1/2pN4/2P1P3/1P4P1/PQ3PBP/R4K2 b - - 2 20",
        "5rr1/4n2k/4q2P/P1P2n2/3B1p2/4pP2/2N1P3/1RR1K2Q w - - 1 49",
        "1r5k/2pq2p1/3p3p/p1pP4/4QP2/PP1R3P/6PK/8 w - - 1 51",
        "q5k1/5ppp/1r3bn1/1B6/P1N2P2/BQ2P1P1/5K1P/8 b - - 2 34",
        "r1b2k1r/5n2/p4q2/1ppn1Pp1/3pp1p1/NP2P3/P1PPBK2/1RQN2R1 w - - 0 22",
        "r1bqk2r/pppp1ppp/5n2/4b3/4P3/P1N5/1PP2PPP/R1BQKB1R w KQkq - 0 5",
        "r1bqr1k1/pp1p1ppp/2p5/8/3N1Q2/P2BB3/1PP2PPP/R3K2n b Q - 1 12",
        "r1bq2k1/p4r1p/1pp2pp1/3p4/1P1B3Q/P2B1N2/2P3PP/4R1K1 b - - 2 19",
        "r4qk1/6r1/1p4p1/2ppBbN1/1p5Q/P7/2P3PP/5RK1 w - - 2 25",
        "r7/6k1/1p6/2pp1p2/7Q/8/p1P2K1P/8 w - - 0 32",
        "r3k2r/ppp1pp1p/2nqb1pn/3p4/4P3/2PP4/PP1NBPPP/R2QK1NR w KQkq - 1 5",
        "3r1rk1/1pp1pn1p/p1n1q1p1/3p4/Q3P3/2P5/PP1NBPPP/4RRK1 w - - 0 12",
        "5rk1/1pp1pn1p/p3Brp1/8/1n6/5N2/PP3PPP/2R2RK1 w - - 2 20",
        "8/1p2pk1p/p1p1r1p1/3n4/8/5R2/PP3PPP/4R1K1 b - - 3 27",
        "8/4pk2/1p1r2p1/p1p4p/Pn5P/3R4/1P3PP1/4RK2 w - - 1 33",
        "8/5k2/1pnrp1p1/p1p4p/P6P/4R1PK/1P3P2/4R3 b - - 1 38",
        "8/8/1p1kp1p1/p1pr1n1p/P6P/1R4P1/1P3PK1/1R6 b - - 15 45",
        "8/8/1p1k2p1/p1prp2p/P2n3P/6P1/1P1R1PK1/4R3 b - - 5 49",
        "8/8/1p4p1/p1p2k1p/P2npP1P/4K1P1/1P6/3R4 w - - 6 54",
        "8/8/1p4p1/p1p2k1p/P2n1P1P/4K1P1/1P6/6R1 b - - 6 59",
        "8/5k2/1p4p1/p1pK3p/P2n1P1P/6P1/1P6/4R3 b - - 14 63",
        "8/1R6/1p1K1kp1/p6p/P1p2P1P/6P1/1Pn5/8 w - - 0 67",
        "1rb1rn1k/p3q1bp/2p3p1/2p1p3/2P1P2N/PP1RQNP1/1B3P2/4R1K1 b - - 4 23",
        "4rrk1/pp1n1pp1/q5p1/P1pP4/2n3P1/7P/1P3PB1/R1BQ1RK1 w - - 3 22",
        "r2qr1k1/pb1nbppp/1pn1p3/2ppP3/3P4/2PB1NN1/PP3PPP/R1BQR1K1 w - - 4 12",
        "2r2k2/8/4P1R1/1p6/8/P4K1N/7b/2B5 b - - 0 55",
        "6k1/5pp1/8/2bKP2P/2P5/p4PNb/B7/8 b - - 1 44",
        "2rqr1k1/1p3p1p/p2p2p1/P1nPb3/2B1P3/5P2/1PQ2NPP/R1R4K w - - 3 25",
        "r1b2rk1/p1q1ppbp/6p1/2Q5/8/4BP2/PPP3PP/2KR1B1R b - - 2 14",
        "6r1/5k2/p1b1r2p/1pB1p1p1/1Pp3PP/2P1R1K1/2P2P2/3R4 w - - 1 36",
        "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
        "2rr2k1/1p4bp/p1q1p1p1/4Pp1n/2PB4/1PN3P1/P3Q2P/2RR2K1 w - f6 0 20",
        "3br1k1/p1pn3p/1p3n2/5pNq/2P1p3/1PN3PP/P2Q1PB1/4R1K1 w - - 0 23",
        "2r2b2/5p2/5k2/p1r1pP2/P2pB3/1P3P2/K1P3R1/7R w - - 23 93"
    };

}

#endif // !UTIL_H