#pragma once

#ifndef MOVE_H
#define MOVE_H

#include "bitboard.h"
#include "types.h"
#include "util.h"

#include <iostream>
#include <sstream>
#include <string>

constexpr i32 FlagEnPassant    = 0b0001 << 12;
constexpr i32 FlagCastle	   = 0b0010 << 12;
constexpr i32 FlagPromotion    = 0b0011 << 12;
constexpr i32 SpecialFlagsMask = 0b0011 << 12;
constexpr i32 FlagPromoHorsie = 0b00 << 14 | FlagPromotion;
constexpr i32 FlagPromoBishop = 0b01 << 14 | FlagPromotion;
constexpr i32 FlagPromoRook   = 0b10 << 14 | FlagPromotion;
constexpr i32 FlagPromoQueen  = 0b11 << 14 | FlagPromotion;

namespace Horsie {

    class Move;
    class Bitboard;
    struct ScoredMove;

    class Move {
    public:
        Move() = default;
        constexpr explicit Move(u16 d) : data(d) {}
        constexpr Move(i32 from, i32 to, i32 flags = 0) : data((u16)(to | (from << 6) | flags)) {}

        constexpr i32 To() const { return i32(data & 0x3F); }
        constexpr i32 From() const { return i32((data >> 6) & 0x3F); }
        constexpr std::pair<i32, i32> Unpack() const { return { From(), To() }; }

        constexpr i32 Data() const { return data; }
        constexpr i32 GetMoveMask() const { return data & 0xFFF; }

        constexpr bool IsEnPassant() const { return ((data & SpecialFlagsMask) == FlagEnPassant); }
        constexpr bool IsCastle() const { return ((data & SpecialFlagsMask) == FlagCastle); }
        constexpr bool IsPromotion() const { return ((data & SpecialFlagsMask) == FlagPromotion); }

        constexpr Piece PromotionTo() const { return Piece(((data >> 14) & 0x3) + 1); }

        constexpr bool IsNull() const { return data == 0; }
        static constexpr Move Null() { return Move(0); }

        constexpr bool operator==(const Move& m) const { return data == m.data; }
        constexpr bool operator!=(const Move& m) const { return data != m.data; }
        constexpr explicit operator bool() const { return data != 0; }

        constexpr i32 CastlingKingSquare() const {
            if (From() < Square::A2) {
                return (To() > From()) ? static_cast<i32>(Square::G1) : static_cast<i32>(Square::C1);
            }

            return (To() > From()) ? static_cast<i32>(Square::G8) : static_cast<i32>(Square::C8);
        }

        constexpr i32 CastlingRookSquare() const {
            if (From() < Square::A2) {
                return (To() > From()) ? static_cast<i32>(Square::F1) : static_cast<i32>(Square::D1);
            }

            return (To() > From()) ? static_cast<i32>(Square::F8) : static_cast<i32>(Square::D8);
        }

        constexpr CastlingStatus RelevantCastlingRight() const {
            if (From() < Square::A2) {
                return (To() > From()) ? CastlingStatus::WK : CastlingStatus::WQ;
            }

            return (To() > From()) ? CastlingStatus::BK : CastlingStatus::BQ;
        }

        std::string ToString(const Bitboard& bb) const {
            const auto [moveFrom, moveTo] = Unpack();
            const auto thisPiece = bb.GetPieceAtIndex(moveFrom);

            if (IsCastle()) {
                return moveTo > moveFrom ? "O-O" : "O-O-O";
            }

            std::stringstream ss;

            const bool isCapture = bb.GetPieceAtIndex(moveTo) != NONE;
            if (thisPiece == PAWN) {
                if (isCapture || IsEnPassant())
                    ss << GetFileChar(GetIndexFile(moveFrom));
            }
            else
                ss << static_cast<char>(toupper(PieceToFEN(thisPiece)));

            const auto multPieces = bb.AttackersTo(moveTo, bb.Occupancy) & bb.Colors[bb.GetColorAtIndex(moveFrom)] & bb.Pieces[thisPiece];

            if (popcount(multPieces) > 1 && thisPiece != PAWN) {
                if ((multPieces & FileBB(moveFrom)) == SquareBB(moveFrom)) {
                    //  If this piece is alone on its file, we only specify the file.
                    ss << (GetFileChar(GetIndexFile(moveFrom)));
                }
                else if ((multPieces & RankBB(moveFrom)) == SquareBB(moveFrom)) {
                    //  If this piece wasn't alone on its file, but is alone on its rank, then include the rank.
                    ss << std::to_string(GetIndexRank(moveFrom) + 1);
                }
                else {
                    //  If neither the rank/file alone could differentiate this move, then we need both the file and rank
                    ss << (GetFileChar(GetIndexFile(moveFrom)));
                    ss << std::to_string(GetIndexRank(moveFrom) + 1);
                }
            }

            if (isCapture || IsEnPassant()) {
                ss << 'x';
            }


            ss << IndexToString(moveTo);

            if (IsPromotion()) {
                ss << '=' << static_cast<char>(toupper(PieceToFEN(PromotionTo())));
            }

            return ss.str();
        }

        constexpr std::string SmithNotation(bool is960) const {
            i32 fx, fy, tx, ty = 0;
            IndexToCoord(From(), fx, fy);
            IndexToCoord(To(), tx, ty);

            if (IsCastle() && !is960) {
                tx = (tx > fx) ? File::FILE_G : File::FILE_C;
            }

            std::string retVal = std::string{ char('a' + fx), char('1' + fy), char('a' + tx), char('1' + ty) };
            if (IsPromotion()) {
                retVal += char(std::tolower(PieceToChar[PromotionTo()]));
            }

            return retVal;
        }

        friend std::ostream& operator<<(std::ostream& os, const Move& m) {
            os << m.SmithNotation(false);
            return os;
        }

        std::string ToString() { return Move::ToString(*this); }
        static std::string ToString(Move m) { return m.SmithNotation(false); }

    private:
        u16 data;
    };


    struct ScoredMove {
        Move move;
        i32 Score;
    };

}


#endif // !MOVE_H