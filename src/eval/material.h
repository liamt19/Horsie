#pragma once

#include "../types.h"
#include "tapered.h"

#include <array>

namespace Horsie::Eval
{
#define S(Mg, Eg) TaperedScore{(Mg), (Eg)}

	namespace PSQT
	{
		constexpr auto Pawn = S(82, 94);
		constexpr auto Horsie = S(337, 281);
		constexpr auto Bishop = S(365, 297);
		constexpr auto Rook = S(477, 512);
		constexpr auto Queen = S(1025, 936);

		constexpr auto King = S(0, 0);

		constexpr std::array<TaperedScore, 7> BaseValues = {
			Pawn,
			Horsie,
			Bishop,
			Rook,
			Queen,
			King,
			TaperedScore{}
		};

		constexpr std::array<TaperedScore, 13> Values = {
			Pawn,
			Pawn,
			Horsie,
			Horsie,
			Bishop,
			Bishop,
			Rook,
			Rook,
			Queen,
			Queen,
			King,
			King,
			TaperedScore{}
		};
	}

#undef S

	namespace PSQT
	{
		extern const std::array<std::array<TaperedScore, 64>, 12> g_pieceSquareTables;
	}

	inline auto pieceSquareValue(i32 type, i32 color, i32 square) {
		const auto piece = type * 2 + color;
		return PSQT::g_pieceSquareTables[piece][square];
	}

	constexpr auto baseValue(i32 type) {
		return PSQT::BaseValues[type];
	}

	constexpr auto pieceValue(i32 type, i32 color) {
		const auto piece = type * 2 + color;
		return PSQT::Values[piece];
	}

	constexpr std::array<i32, 12> Phase = {
		0, 0, 1, 1, 2, 2, 2, 2, 4, 4, 0, 0,
	};

	constexpr i32 PHASE_MAX = 24;

	struct MaterialScore {
		TaperedScore score{};
		i32 phase{};

		constexpr auto subAdd(i32 type, i32 color, i32 src, i32 dst) {
			score -= pieceSquareValue(type, color, src);
			score += pieceSquareValue(type, color, dst);
		}

		constexpr auto add(i32 type, i32 color, i32 square) {
			const auto piece = type * 2 + color;
			phase += Phase[piece];
			score += pieceSquareValue(type, color, square);
		}

		constexpr auto sub(i32 type, i32 color, i32 square) {
			const auto piece = type * 2 + color;
			phase -= Phase[piece];
			score -= pieceSquareValue(type, color, square);
		}

		inline auto get() const {
			const auto a = score.endgame();
			const auto b = score.midgame();
			const auto t = std::min(phase, PHASE_MAX);
			return (a * (PHASE_MAX - t) + b * t) / PHASE_MAX;
		}
	};
}