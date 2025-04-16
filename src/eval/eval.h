#pragma once

#include "../position.h"

namespace Horsie::Eval
{
	inline auto GetEvaluation(const Position& pos) {
		const auto eval = pos.GetMaterialScore();
		return std::clamp(eval, -ScoreWin + 1, ScoreWin - 1);
	}
}