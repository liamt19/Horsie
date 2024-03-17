#pragma once

#ifndef BITBOARD_H
#define BITBOARD_H

#include <string>
#include <cassert>

#include "util.h"

#define Pawn PAWN
#define Knight HORSIE
#define Bishop BISHOP
#define Rook ROOK
#define Queen QUEEN
#define King KING

namespace Horsie {

	class Bitboard
	{
	public:
		ulong Pieces[6];
		ulong Colors[2];
		int PieceTypes[64];

		ulong Occupancy = 0;

		Bitboard();
		~Bitboard() = default;

		constexpr int GetColorAtIndex(int idx) const { return ((Colors[Color::WHITE] & SquareBB(idx)) != 0) ? Color::WHITE : Color::BLACK; }
		constexpr int GetPieceAtIndex(int idx) const { return PieceTypes[idx]; }
		constexpr bool IsColorSet(int pc, int idx) const { return (Colors[pc] & SquareBB(idx)) != 0; }
		constexpr ulong KingMask(int pc) const { return Colors[pc] & Pieces[Piece::KING]; }
		constexpr bool Occupied(int idx) const { return PieceTypes[idx] != Piece::NONE; }


		void Reset();
		void AddPiece(int idx, int pc, int pt);
		void RemovePiece(int idx, int pc, int pt);
		void MoveSimple(int from, int to, int pieceColor, int pieceType);
		int KingIndex(int pc) const;
		int MaterialCount(int pc, bool excludePawns) const;
		ulong BlockingPieces(int pc, ulong* pinners, ulong* xrayers) const;
		ulong AttackersTo(int idx, ulong occupied) const;
		ulong AttackersToMajors(int idx, ulong occupied) const;
		ulong AttackMask(int idx, int pc, int pt, ulong occupied) const;
		ulong AttackMask(int pc, int pt) const;
	};
}



#endif // !BITBOARD_H