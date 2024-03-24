#pragma once

#ifndef MOVE_H
#define MOVE_H

#include "types.h"
#include "util.h"


constexpr int FlagEnPassant = 0b000001 << 14;
constexpr int FlagCastle = 0b000010 << 14;
constexpr int FlagPromotion = 0b000011 << 14;
constexpr int SpecialFlagsMask = 0b000011 << 14;

namespace Horsie {

    class Move;
    struct ScoredMove;

    class Move {
    public:
        Move() = default;
        constexpr explicit Move(ushort d) : data(d) {}
        constexpr Move(int from, int to) : data((ushort)((from << 6) + to)) {}

        static constexpr Move make(int from, int to) {
            return Move((int)to | ((int)from << 6));
        }
        
        static constexpr Move make(int from, int to, Piece promotionTo) {
            return Move((int)to | ((int)from << 6) | ((promotionTo - 1) << 12) | FlagPromotion);
        }

        constexpr void SetNew(int from, int to) { data = ((int)to | ((int)from << 6)); }
        constexpr void SetNew(int from, int to, int promotionTo) { data = ((int)to | ((int)from << 6) | ((promotionTo - 1) << 12) | FlagPromotion); }

        constexpr int From() const {
            //assert(IsOK());
            return int((data >> 6) & 0x3F);
        }

        constexpr int To() const {
            //assert(IsOK());
            return int(data & 0x3F);
        }

        constexpr int Data() const { return data; }
        constexpr int GetMoveMask() const { return data & 0xFFF; }

        constexpr void SetEnPassant() { data |= FlagEnPassant; }
        constexpr void SetCastle() { data |= FlagCastle; }

        constexpr bool IsEnPassant() const { return ((data & SpecialFlagsMask) == FlagEnPassant); }
        constexpr bool IsCastle() const { return ((data & SpecialFlagsMask) == FlagCastle); }
        constexpr bool IsPromotion() const { return ((data & SpecialFlagsMask) == FlagPromotion); }

        constexpr Piece PromotionTo() const { return Piece(((data >> 12) & 0x3) + 1); }

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

#ifdef DEBUG
            retVal = "[" + retVal;
            retVal += (IsEnPassant() ? " EP" : "");
            retVal += (IsCastle() ? " Castle" : "");
            return retVal + "]";
#endif
            return retVal;
        }

        friend std::ostream& operator<<(std::ostream& os, const Move& m)
        {
            os << m.SmithNotation(false);
            return os;
        }


        static std::string ToString(Move m) {
            return m.SmithNotation(false);
        }

    private:
        ushort data;
    };


    struct ScoredMove {
        Move Move;
        int Score;
    };


}

#endif // !MOVE_H