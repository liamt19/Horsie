#pragma once

#ifndef MOVE_H
#define MOVE_H

#include "types.h"
#include "util.h"

#include <string>  
#include <iostream> 
#include <sstream>   


constexpr int FlagEnPassant = 0b0001 << 12;
constexpr int FlagCastle    = 0b0010 << 12;
constexpr int FlagPromotion = 0b0011 << 12;
constexpr int SpecialFlagsMask  = 0b0011 << 12;
constexpr int FlagPromoKnight   = 0b00 << 14 | FlagPromotion;
constexpr int FlagPromoBishop   = 0b01 << 14 | FlagPromotion;
constexpr int FlagPromoRook     = 0b10 << 14 | FlagPromotion;
constexpr int FlagPromoQueen    = 0b11 << 14 | FlagPromotion;

namespace Horsie {

    class Move;
    struct ScoredMove;

    class Move {
    public:
        Move() = default;
        constexpr explicit Move(ushort d) : data(d) {}
        constexpr Move(int from, int to, int flags = 0) : data((ushort)(to | (from << 6) | flags)) {}

        constexpr int To() const { return int(data & 0x3F); }
        constexpr int From() const { return int((data >> 6) & 0x3F); }

        constexpr int Data() const { return data; }
        constexpr int GetMoveMask() const { return data & 0xFFF; }

        constexpr bool IsEnPassant() const { return ((data & SpecialFlagsMask) == FlagEnPassant); }
        constexpr bool IsCastle() const {    return ((data & SpecialFlagsMask) == FlagCastle); }
        constexpr bool IsPromotion() const { return ((data & SpecialFlagsMask) == FlagPromotion); }

        constexpr Piece PromotionTo() const { return Piece(((data >> 14) & 0x3) + 1); }

        constexpr bool IsNull() const { return data != 0; }
        static constexpr Move Null() { return Move(0); }

        constexpr bool operator==(const Move& m) const { return data == m.data; }
        constexpr bool operator!=(const Move& m) const { return data != m.data; }
        constexpr explicit operator bool() const { return data != 0; }

        constexpr bool IsOK() const { return Null().data != data; }

        constexpr int CastlingKingSquare() const {
            if (From() < (int)Square::A2) {
                return (To() > From()) ? (int)Square::G1 : (int)Square::C1;
            }

            return (To() > From()) ? (int)Square::G8 : (int)Square::C8;
        }

        constexpr int CastlingRookSquare() const {
            if (From() < (int)Square::A2) {
                return (To() > From()) ? (int)Square::F1 : (int)Square::D1;
            }

            return (To() > From()) ? (int)Square::F8 : (int)Square::D8;
        }

        constexpr CastlingStatus RelevantCastlingRight() const {
            if (From() < (int)Square::A2) {
                return (To() > From()) ? CastlingStatus::WK : CastlingStatus::WQ;
            }

            return (To() > From()) ? CastlingStatus::BK : CastlingStatus::BQ;
        }

        std::string SmithNotation(bool is960) const {
            //int fx = From() % 8;
            //int fy = From() / 8;
            //int tx = To() % 8;
            //int ty = To() / 8;

            int fx, fy, tx, ty = 0;
            IndexToCoord((int) From(), fx, fy);
            IndexToCoord((int) To(), tx, ty);


            if (IsCastle() && !is960)
            {
                tx = (tx > fx) ? File::FILE_G : File::FILE_C;
            }


            std::string retVal = std::string{char('a' + fx), char('1' + fy)} + std::string{char('a' + tx), char('1' + ty)};
            if (IsPromotion()) {
                retVal += char(std::tolower(PieceToChar[PromotionTo()]));
            }

            return retVal;
        }

        friend std::ostream& operator<<(std::ostream& os, const Move& m)
        {
            os << m.SmithNotation(false);
            return os;
        }

        std::string ToString() { return Move::ToString(*this); }

        static std::string ToString(Move m) {
            return m.SmithNotation(false);
        }

    private:
        ushort data;
    };


    struct ScoredMove {
        Move move;
        int Score;
    };



    inline std::string MoveListToString(const ScoredMove* moves, int size) {
        std::stringstream buffer;
        for (int i = 0; i < size; i++)
        {
			buffer << Move::ToString(moves[i].move) << ": " << moves[i].Score << "\n";
		}

		return buffer.str();
	}


}

#endif // !MOVE_H