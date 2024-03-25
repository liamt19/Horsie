#pragma once

#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <cassert>

#include "util.h"
#include "enums.h"

using ushort = uint16_t;
using ulong = uint64_t;
using sbyte = int8_t;

// Predefined macros hell:
//
// __GNUC__                Compiler is GCC, Clang or ICX
// __clang__               Compiler is Clang or ICX
// __INTEL_LLVM_COMPILER   Compiler is ICX
// _MSC_VER                Compiler is MSVC
// _WIN32                  Building on Windows (any)
// _WIN64                  Building on Windows 64 bit

#if defined(__GNUC__) && (__GNUC__ < 9 || (__GNUC__ == 9 && __GNUC_MINOR__ <= 2)) \
      && defined(_WIN32) && !defined(__clang__)
#define ALIGNAS_ON_STACK_VARIABLES_BROKEN
#endif

#define ASSERT_ALIGNED(ptr, alignment) assert(reinterpret_cast<uintptr_t>(ptr) % alignment == 0)

#if defined(_WIN64) && defined(_MSC_VER)  // No Makefile used
#include <intrin.h>                   // Microsoft header for _BitScanForward64()
#define IS_64BIT
#endif

#if defined(USE_POPCNT) && defined(_MSC_VER)
#include <nmmintrin.h>  // Microsoft header for _mm_popcnt_u64()
#endif

#if !defined(NO_PREFETCH) && defined(_MSC_VER)
#include <xmmintrin.h>  // Microsoft header for _mm_prefetch()
#endif

#if defined(USE_PEXT)
#include <immintrin.h>  // Header for _pext_u64() intrinsic
#define pext(b, m) _pext_u64(b, m)
#else
#define pext(b, m) 0
#endif

namespace Horsie {

constexpr ulong FileABB = 0x0101010101010101ULL;
constexpr ulong FileBBB = FileABB << 1;
constexpr ulong FileCBB = FileABB << 2;
constexpr ulong FileDBB = FileABB << 3;
constexpr ulong FileEBB = FileABB << 4;
constexpr ulong FileFBB = FileABB << 5;
constexpr ulong FileGBB = FileABB << 6;
constexpr ulong FileHBB = FileABB << 7;

constexpr ulong Rank1BB = 0xFF;
constexpr ulong Rank2BB = Rank1BB << (8 * 1);
constexpr ulong Rank3BB = Rank1BB << (8 * 2);
constexpr ulong Rank4BB = Rank1BB << (8 * 3);
constexpr ulong Rank5BB = Rank1BB << (8 * 4);
constexpr ulong Rank6BB = Rank1BB << (8 * 5);
constexpr ulong Rank7BB = Rank1BB << (8 * 6);
constexpr ulong Rank8BB = Rank1BB << (8 * 7);

constexpr auto InitialFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr auto KiwiFEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10";

#define N_TABS(n) \
        for (int _n_ = 0; _n_ < n; _n_++) std::cout << "\t";

#define ENABLE_INCR_OPERATORS_ON(T) \
        inline T& operator++(T& d) { return d = T(int(d) + 1); } \
        inline T& operator--(T& d) { return d = T(int(d) - 1); }

ENABLE_INCR_OPERATORS_ON(Piece)
ENABLE_INCR_OPERATORS_ON(Square)
ENABLE_INCR_OPERATORS_ON(File)
ENABLE_INCR_OPERATORS_ON(Rank)

#undef ENABLE_INCR_OPERATORS_ON

}

namespace {

constexpr Direction operator+(Direction d1, Direction d2) { return Direction(int(d1) + int(d2)); }
constexpr Direction operator*(int i, Direction d) { return Direction(i * int(d)); }

// Additional operators to add a Direction to a Square
constexpr Square operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
inline Square& operator+=(Square& s, Direction d) { return s = s + d; }
inline Square& operator-=(Square& s, Direction d) { return s = s - d; }
// Toggle color
constexpr Color operator~(Color c) { return Color(c ^ 1); }
constexpr int Not(int c) { return c ^ 1; }
constexpr Color Not(Color c) { return Color(int(c) ^ 1); }

//constexpr CastlingStatus operator~(CastlingStatus l) { return ~l; }
//constexpr CastlingStatus operator&(CastlingStatus l, CastlingStatus r) { return CastlingStatus(int(l) & int(r)); }
//constexpr CastlingStatus operator|(CastlingStatus l, CastlingStatus r) { return CastlingStatus(int(l) | int(r)); }
//constexpr CastlingStatus& operator&=(CastlingStatus& l, CastlingStatus r) { return l &= r; }
//constexpr CastlingStatus& operator|=(CastlingStatus& l, CastlingStatus r) { return l |= r; }

constexpr Square operator^(Square s1, Square s2) { return Square(int(s1) ^ int(s2)); }
constexpr Square operator-(Square s1, Square s2) { return Square(int(s1) - int(s2)); }
constexpr Square operator+(Square s1, Square s2) { return Square(int(s1) + int(s2)); }

constexpr Square operator^(Square s1, int s2) { return Square(int(s1) ^ s2); }
constexpr Square operator-(Square s1, int s2) { return Square(int(s1) - s2); }
constexpr Square operator+(Square s1, int s2) { return Square(int(s1) + s2); }
constexpr Square operator<<(Square s1, int s2) { return Square(int(s1) << s2); }
constexpr Square operator%(Square s1, int s2) { return Square(int(s1) % s2); }
constexpr Square operator/(Square s1, int s2) { return Square(int(s1) / s2); }


#ifdef USE_PEXT
constexpr bool HasPext = true;
#else
constexpr bool HasPext = false;
#endif

    inline void prefetch(void* b) {
#if defined(_MSC_VER)
        _mm_prefetch((char*)b, _MM_HINT_T0);
#else
        __builtin_prefetch(b);
#endif
    }


    inline int popcount(ulong b) {

#if defined(_MSC_VER)
        return int(__popcnt64(b));
#else
        return __builtin_popcountll(b);
#endif
    }

    // Returns the least significant bit in a non-zero bitboard.
    inline int lsb(ulong b) {
        assert(b);

#if defined(__GNUC__)  // GCC, Clang, ICX
        return int(__builtin_ctzll(b));
#else
        unsigned long idx;
        _BitScanForward64(&idx, b);
        return int(idx);
#endif
    }

    // Returns the most significant bit in a non-zero bitboard.
    inline int msb(ulong b) {
        assert(b);

#if defined(__GNUC__)  // GCC, Clang, ICX
        return int(63 ^ __builtin_clzll(b));
#else
        unsigned long idx;
        _BitScanReverse64(&idx, b);
        return int(idx);
#endif
    }

    // Finds and clears the least significant bit in a non-zero bitboard.
    inline int poplsb(ulong& b) {
        assert(b);
        int s = lsb(b);
        b &= b - 1;
        return int(s);
    }
}

namespace Horsie {


#define AlignedFree _aligned_free
#define AlignedAllocZeroed _aligned_malloc
#define MemClear memset
#define CopyBlock memcpy

    constexpr size_t AllocAlignment = 64;

    constexpr ulong SquareBB(int s) { return (1ULL << s); }
    constexpr ulong SquareBB(Square s) { return SquareBB((int)s); }
    constexpr bool MoreThanOne(ulong b) { return b & (b - 1); }

    constexpr ulong RankBB(Rank r) { return Rank1BB << (8 * r); }
    constexpr ulong FileBB(File f) { return FileABB << f; }

    constexpr Rank GetIndexRank(int s) { return Rank(int(s) >> 3); }
    constexpr File GetIndexFile(int s) { return File((int)s & 7); }

    constexpr ulong RankBB(int s) { return RankBB(GetIndexRank(s)); }
    constexpr ulong FileBB(int s) { return FileBB(GetIndexFile(s)); }

    constexpr char GetFileChar(int fileNumber) { return (char)(97 + fileNumber); }
    constexpr int GetFileInt(char fileLetter) { return fileLetter - 97; }

    constexpr bool IsOK(int s) { return s >= (int)Square::A1 && s <= (int)Square::H8; }

    constexpr int ShiftUpDir(int c) { return c == WHITE ? NORTH : SOUTH; };

    template<Direction D>
    constexpr ulong Shift(ulong b) {
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

    constexpr ulong Forward(int c, ulong b) { return c == WHITE ? Shift<NORTH>(b) : Shift<SOUTH>(b); };


    inline ulong  operator&(ulong b, Square s) { return b & SquareBB(s); }
    inline ulong  operator|(ulong b, Square s) { return b | SquareBB(s); }
    inline ulong  operator^(ulong b, Square s) { return b ^ SquareBB(s); }
    inline ulong& operator|=(ulong& b, Square s) { return b |= SquareBB(s); }
    inline ulong& operator^=(ulong& b, Square s) { return b ^= SquareBB(s); }

    inline ulong operator&(Square s, ulong b) { return b & s; }
    inline ulong operator|(Square s, ulong b) { return b | s; }
    inline ulong operator^(Square s, ulong b) { return b ^ s; }

    inline ulong operator|(Square s1, Square s2) { return SquareBB(s1) | s2; }


    constexpr bool DirectionOK(Square sq, Direction dir)
    {
        if (sq + dir < Square::A1 || sq + dir > Square::H8)
        {
            //  Make sure we aren't going off the board.
            return false;
        }

        //  The rank and file of (sq + dir) should only change by at most 2 for knight moves,
        //  and 1 for bishop or rook moves.
        int rankDistance = std::abs(GetIndexRank((int)sq) - GetIndexRank((int)sq + dir));
        int fileDistance = std::abs(GetIndexFile((int)sq) - GetIndexFile((int)sq + dir));
        return std::max(rankDistance, fileDistance) <= 2;
    }

    constexpr void IndexToCoord(int index, int& x, int& y) {
        x = index % 8;
        y = index / 8;
    }

    constexpr int CoordToIndex(int x, int y) { return (y * 8) + x; }

    inline const std::string ColorToString(Color color) { return color == WHITE ? "White" : "Black"; }
    inline int StringToColor(const std::string& color) { return color == "White" ? WHITE : color == "Black" ? BLACK : COLOR_NB; }

    inline const std::string PieceToString(Piece n) {
        switch (n) {
        case PAWN: return "Pawn";
        case HORSIE: return "Horsie";
        case BISHOP: return "Bishop";
        case ROOK: return "Rook";
        case QUEEN: return "Queen";
        case KING: return "King";
        default: return "None";
        }
    }

    inline int StringToPiece(const std::string& pieceName) {
        if (pieceName == "Pawn") return PAWN;
        if (pieceName == "Horsie") return HORSIE;
        if (pieceName == "Bishop") return BISHOP;
        if (pieceName == "Rook") return ROOK;
        if (pieceName == "Queen") return QUEEN;
        if (pieceName == "King") return KING;
        return NONE;
    }

    inline int MakePiece(int pc, int pt) {
        return (pc * 2) + pt;
    }


    struct Accumulator;
    struct StateInfo {
    public:

        ulong CheckSquares[PIECE_NB];
        int KingSquares[2];
        ulong BlockingPieces[2];
        ulong Pinners[2];
        ulong Xrays[2];
        ulong Hash = 0;
        ulong Checkers = 0;
        CastlingStatus CastleStatus = CastlingStatus::None;
        int HalfmoveClock = 0;
        int EPSquare = EP_NONE;
        int CapturedPiece = Piece::NONE;

        Accumulator* Accumulator;
    };

    constexpr static int StateCopySize = sizeof(StateInfo) - sizeof(ulong);
}


#endif // !TYPES_H