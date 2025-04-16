#pragma once

#include "../defs.h"
#include "../types.h"
#include <bit>

#include <cassert>
#include <cstring>
#include <limits>

namespace Horsie::Eval
{
	struct TaperedScore {
		i32 m_score;
		explicit constexpr TaperedScore(i32 score) : m_score{ score } {}
		constexpr TaperedScore() : m_score{} {}

		constexpr TaperedScore(Score midgame, Score endgame) : m_score{ static_cast<i32>(static_cast<u32>(endgame) << 16) + midgame } {
			assert(std::numeric_limits<i16>::min() <= midgame && std::numeric_limits<i16>::max() >= midgame);
			assert(std::numeric_limits<i16>::min() <= endgame && std::numeric_limits<i16>::max() >= endgame);
		}

		inline Score midgame() const {
			const auto mg = static_cast<u16>(m_score);
			return static_cast<Score>(std::bit_cast<i16>(mg));
		}

		inline Score endgame() const {
			const auto eg = static_cast<u16>(static_cast<u32>(m_score + 0x8000) >> 16);
			return static_cast<Score>(std::bit_cast<i16>(eg));
		}

		constexpr auto operator+(const TaperedScore& other) const {
			return TaperedScore{ m_score + other.m_score };
		}

		constexpr auto operator+=(const TaperedScore& other) {
			m_score += other.m_score;
			return *this;
		}

		constexpr auto operator-(const TaperedScore& other) const {
			return TaperedScore{ m_score - other.m_score };
		}

		constexpr auto operator-=(const TaperedScore& other) {
			m_score -= other.m_score;
			return *this;
		}

		constexpr auto operator-() const {
			return TaperedScore{ -m_score };
		}
	};
}