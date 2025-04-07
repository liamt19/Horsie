#pragma once

#include "defs.h"

namespace Horsie::WDL {

    const double EvalNormalization = 231;

    std::pair<i32, i32> MaterialModel(i32 score, i32 mat);
    i32 NormalizeScore(i32 score);

}
