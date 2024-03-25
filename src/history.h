#pragma once

#ifndef HISTORY_H
#define HISTORY_H


namespace Horsie {

    constexpr int HistoryClamp = 16384;

    struct PieceToHistory {
        short history[PIECE_NB * 2][SQUARE_NB];

        void Clear() {
            for (size_t i = 0; i < PIECE_NB * 2; i++)
            {
                for (size_t j = 0; j < SQUARE_NB; j++)
                {
                    history[i][j] = -50;
                }
            }
        }
    };

    struct ContinuationHistory {
        PieceToHistory Histories[PIECE_NB * 2][SQUARE_NB];
        
        void Clear() {
            for (size_t i = 0; i < PIECE_NB * 2; i++)
            {
                for (size_t j = 0; j < SQUARE_NB; j++)
                {
                    Histories[i][j].Clear();
                }
            }
        }
    };

    struct HistoryTable {
    public:
        short MainHistory[COLOR_NB][SQUARE_NB * SQUARE_NB];

        //  Index with [pc][pt][toSquare][capturedType]
        short CaptureHistory[COLOR_NB][PIECE_NB][SQUARE_NB][PIECE_NB];

        ContinuationHistory Continuations[2][2];

        void Clear() {
            for (size_t i = 0; i < COLOR_NB; i++)
            {
                for (size_t j = 0; j < SQUARE_NB * SQUARE_NB; j++)
                {
					MainHistory[i][j] = 0;
				}
			}

            for (size_t i = 0; i < COLOR_NB; i++)
            {
                for (size_t j = 0; j < PIECE_NB; j++)
                {
                    for (size_t k = 0; k < SQUARE_NB; k++)
                    {
                        for (size_t l = 0; l < PIECE_NB; l++)
                        {
							CaptureHistory[i][j][k][l] = 0;
						}
					}
				}
			}

            for (size_t i = 0; i < 2; i++)
            {
                for (size_t j = 0; j < 2; j++)
                {
					Continuations[i][j].Clear();
				}
			}
		}
    };

}


#endif