#pragma once

#ifndef HISTORY_H
#define HISTORY_H


namespace Horsie {

    constexpr int HistoryClamp = 16384;

    struct PieceToHistory {
        short history[PIECE_NB * 2][SQUARE_NB];
    };

    struct ContinuationHistory {
        PieceToHistory Histories[PIECE_NB * 2][SQUARE_NB];
    };

    class HistoryTable {
    public:
        short MainHistory[COLOR_NB][SQUARE_NB * SQUARE_NB];

        //  Index with [pc][pt][toSquare][capturedType]
        short CaptureHistory[COLOR_NB][PIECE_NB][SQUARE_NB][PIECE_NB];

        ContinuationHistory Continuations[2][2];
    };

}


#endif