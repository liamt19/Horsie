#pragma once

#include "defs.h"

enum Piece : i32 {
    PAWN = 0,
    HORSIE,
    BISHOP,
    ROOK,
    QUEEN,
    KING,

    NONE = 6,
    PIECE_NB = 6
};

enum Color : i32 {
    WHITE = 0,
    BLACK,

    COLOR_NB = 2
};

enum class Square : i32 {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,

    EP_NONE = 0,
    SQUARE_NB = 64,
};

constexpr bool operator<(Square l, i32 r) { return static_cast<i32>(l) < r; }
constexpr bool operator<(i32 l, Square r) { return l < static_cast<i32>(r); }
constexpr bool operator>(Square l, i32 r) { return static_cast<i32>(l) > r; }
constexpr bool operator>(i32 l, Square r) { return l > static_cast<i32>(r); }

constexpr i32 EP_NONE = static_cast<i32>(Square::EP_NONE);
constexpr i32 SQUARE_NB = static_cast<i32>(Square::SQUARE_NB);

enum File : i32 {
    FILE_A = 0,
    FILE_B,
    FILE_C,
    FILE_D,
    FILE_E,
    FILE_F,
    FILE_G,
    FILE_H,
    FILE_NB
};

enum Rank : i32 {
    RANK_1 = 0,
    RANK_2,
    RANK_3,
    RANK_4,
    RANK_5,
    RANK_6,
    RANK_7,
    RANK_8,
    RANK_NB
};

enum class CastlingStatus : i32 {
    None = 0,
    WK = 1,
    WQ = 2,
    BK = 4,
    BQ = 8,

    White = WK | WQ,
    Black = BK | BQ,

    Kingside = WK | BK,
    Queenside = WQ | BQ,

    All = WK | WQ | BK | BQ,
};

enum Direction : i32 {
    NORTH = 8,
    EAST = 1,
    SOUTH = -NORTH,
    WEST = -EAST,

    NORTH_EAST = NORTH + EAST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST,
    NORTH_WEST = NORTH + WEST
};

enum TTNodeType {
    Invalid,
    /// <summary>
    /// Upper Bound
    /// </summary>
    Beta,
    /// <summary>
    /// Lower bound
    /// </summary>
    Alpha,
    Exact = Beta | Alpha
};

constexpr CastlingStatus operator~(CastlingStatus l) { return CastlingStatus(~i32(l)); }
constexpr CastlingStatus operator&(CastlingStatus l, CastlingStatus r) { return CastlingStatus(i32(l) & i32(r)); }
constexpr CastlingStatus operator|(CastlingStatus l, CastlingStatus r) { return CastlingStatus(i32(l) | i32(r)); }
constexpr CastlingStatus& operator&=(CastlingStatus& l, CastlingStatus r) { return l = CastlingStatus(i32(l) & i32(r)); }
constexpr CastlingStatus& operator|=(CastlingStatus& l, CastlingStatus r) { return l = CastlingStatus(i32(l) | i32(r)); }
