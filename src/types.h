#pragma once

#include "defs.h"
#include "enums.h"

#include <array>
#include <bit>
#include <cmath>
#include <string>
#include <utility>

#if defined(_MSC_VER)
#include <xmmintrin.h>
#endif

#if defined(USE_PEXT)
#include <immintrin.h>
#define pext(b, m) _pext_u64(b, m)
#else
#define pext(b, m) 0
#endif


namespace Horsie {

    constexpr auto InitialFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    constexpr auto EngVersion = "1.0.21";

    constexpr u64 FileABB = 0x0101010101010101ULL;
    constexpr u64 FileBBB = FileABB << 1;
    constexpr u64 FileCBB = FileABB << 2;
    constexpr u64 FileDBB = FileABB << 3;
    constexpr u64 FileEBB = FileABB << 4;
    constexpr u64 FileFBB = FileABB << 5;
    constexpr u64 FileGBB = FileABB << 6;
    constexpr u64 FileHBB = FileABB << 7;

    constexpr u64 Rank1BB = 0xFF;
    constexpr u64 Rank2BB = Rank1BB << (8 * 1);
    constexpr u64 Rank3BB = Rank1BB << (8 * 2);
    constexpr u64 Rank4BB = Rank1BB << (8 * 3);
    constexpr u64 Rank5BB = Rank1BB << (8 * 4);
    constexpr u64 Rank6BB = Rank1BB << (8 * 5);
    constexpr u64 Rank7BB = Rank1BB << (8 * 6);
    constexpr u64 Rank8BB = Rank1BB << (8 * 7);

#define ENABLE_INCR_OPERATORS_ON(T) \
        inline T& operator++(T& d) { return d = T(i32(d) + 1); } \
        inline T& operator--(T& d) { return d = T(i32(d) - 1); }

    ENABLE_INCR_OPERATORS_ON(Piece)
    ENABLE_INCR_OPERATORS_ON(Square)
    ENABLE_INCR_OPERATORS_ON(File)
    ENABLE_INCR_OPERATORS_ON(Rank)

#undef ENABLE_INCR_OPERATORS_ON

}

namespace {

    constexpr Direction operator+(Direction d1, Direction d2) { return Direction(i32(d1) + i32(d2)); }

    constexpr Square operator+(Square s, Direction d) { return Square(i32(s) + i32(d)); }
    constexpr Square operator-(Square s, Direction d) { return Square(i32(s) - i32(d)); }
    constexpr Square& operator+=(Square& s, Direction d) { return s = s + d; }
    constexpr Square& operator-=(Square& s, Direction d) { return s = s - d; }

    constexpr i32 Not(i32 c) { return c ^ 1; }
    constexpr Color Not(Color c) { return Color(i32(c) ^ 1); }


#ifdef USE_PEXT
    constexpr bool UsePext = true;
#else
    constexpr bool UsePext = false;
#endif

    inline void prefetch(void* b) {
#if defined(_MSC_VER)
        _mm_prefetch((char*)b, _MM_HINT_T0);
#else
        __builtin_prefetch(b);
#endif
    }

    constexpr inline i32 popcount(u64 b) {
        return std::popcount(b);
    }

    constexpr inline i32 lsb(u64 b) {
        return std::countr_zero(b);
    }

    constexpr inline i32 poplsb(u64& b) {
        const auto s = lsb(b);
        b &= b - 1;
        return s;
    }

    constexpr bool MoreThanOne(u64 b) { return b & (b - 1); }
}

namespace Horsie {

    constexpr u64 SquareBB(i32 s) { return (1ULL << s); }
    constexpr u64 SquareBB(Square s) { return SquareBB(static_cast<i32>(s)); }
    
    constexpr Square OrientSquare(Square s, Color c) { return Square(static_cast<i32>(s) ^ i32(56 * c)); }
    constexpr Square OrientSquare(Square s, bool b) { return Square(static_cast<i32>(s) ^ (56 * b)); }

    constexpr u64 RankBB(Rank r) { return Rank1BB << (8 * r); }
    constexpr u64 FileBB(File f) { return FileABB << f; }

    constexpr Rank GetIndexRank(i32 s) { return Rank(s >> 3); }
    constexpr File GetIndexFile(i32 s) { return File(s & 7); }

    constexpr u64 RankBB(i32 s) { return RankBB(GetIndexRank(s)); }
    constexpr u64 FileBB(i32 s) { return FileBB(GetIndexFile(s)); }

    constexpr char GetFileChar(i32 fileNumber) { return char(97 + fileNumber); }
    constexpr i32 GetFileInt(char fileLetter) { return fileLetter - 97; }

    constexpr u64 Shift(Direction D, u64 b) {
        return D == NORTH         ? b << 8
             : D == SOUTH         ? b >> 8
             : D == NORTH + NORTH ? b << 16
             : D == SOUTH + SOUTH ? b >> 16
             : D == EAST          ? (b & ~FileHBB) << 1
             : D == WEST          ? (b & ~FileABB) >> 1
             : D == NORTH_EAST    ? (b & ~FileHBB) << 9
             : D == NORTH_WEST    ? (b & ~FileABB) << 7
             : D == SOUTH_EAST    ? (b & ~FileHBB) >> 7
             : D == SOUTH_WEST    ? (b & ~FileABB) >> 9
                                  : 0;
    }

    template<Direction D>
    constexpr u64 Shift(u64 b) { return Shift(D, b); }

    constexpr u64 Forward(i32 c, u64 b) { return c == WHITE ? Shift<NORTH>(b) : Shift<SOUTH>(b); };
    constexpr i32 ShiftUpDir(i32 c) { return c == WHITE ? NORTH : SOUTH; };

    constexpr std::pair<i32, i32> UnpackIndex(i32 index) { return { index % 8, index / 8 }; }

    constexpr i32 MakeIndex(i32 x, i32 y) { return (y * 8) + x; }

    constexpr i32 MakePiece(i32 pc, i32 pt) { return (pc * 6) + pt; }

    inline const std::string ColorToString(Color color) { return color == WHITE ? "White" : "Black"; }
    inline i32 StringToColor(const std::string& color) { return color == "White" ? WHITE : color == "Black" ? BLACK : COLOR_NB; }

    inline const std::string IndexToString(i32 sq) {
        const auto [x, y] = UnpackIndex(sq);
        return std::string{ char('a' + x), char('1' + y) };
    }

    inline const std::string PieceToString(Piece n) {
        constexpr std::array<std::string, 7> arr = { "Pawn", "Horsie", "Bishop", "Rook", "Queen", "King", "None" };
        return arr[n];
    }


    inline u64  operator&(u64 b, Square s) { return b & SquareBB(s); }
    inline u64  operator|(u64 b, Square s) { return b | SquareBB(s); }
    inline u64  operator^(u64 b, Square s) { return b ^ SquareBB(s); }
    inline u64& operator|=(u64& b, Square s) { return b |= SquareBB(s); }
    inline u64& operator^=(u64& b, Square s) { return b ^= SquareBB(s); }

    inline u64 operator&(Square s, u64 b) { return b & s; }
    inline u64 operator|(Square s, u64 b) { return b | s; }
    inline u64 operator^(Square s, u64 b) { return b ^ s; }

    inline u64 operator|(Square s1, Square s2) { return SquareBB(s1) | s2; }


    namespace NNUE {
        struct Accumulator; 
    }

    struct alignas(32) StateInfo {
    public:

        u64 CheckSquares[PIECE_NB];
        u64 BlockingPieces[2];
        u64 Pinners[2];
        i32 KingSquares[2];
        u64 Checkers = 0;
        u64 Hash = 0;
        u64 PawnHash = 0;
        u64 NonPawnHash[2];
        i32 HalfmoveClock = 0;
        i32 EPSquare = EP_NONE;
        i32 CapturedPiece = Piece::NONE;
        i32 PliesFromNull = 0;
        CastlingStatus CastleStatus = CastlingStatus::None;

        NNUE::Accumulator* accumulator;
    };

    constexpr static i32 StateCopySize = sizeof(StateInfo) - sizeof(u64);
}
