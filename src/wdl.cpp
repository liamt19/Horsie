
#include "wdl.h"

#include "defs.h"
#include "util.h"

#include <algorithm>
#include <cmath>

namespace Horsie::WDL {

    constexpr double as[] = { -148.62396310, 471.71343921, -582.36906088, 508.53478267 };
    constexpr double bs[] = { -13.56385731, 59.30588472, -76.33890789, 97.04639897 };

    std::pair<i32, i32> MaterialModel(i32 score, i32 mat) {
        const auto m = std::clamp(mat, 17, 78) / 58;
        const auto x = std::clamp(score, -4000, 4000);

        auto a = (((as[0] * m + as[1]) * m + as[2]) * m) + as[3];
        auto b = (((bs[0] * m + bs[1]) * m + bs[2]) * m) + bs[3];

        auto win  = std::round(1000.0 / (1 + std::exp((a - x) / b)));
        auto loss = std::round(1000.0 / (1 + std::exp((a + x) / b)));

        return { static_cast<i32>(win), static_cast<i32>(loss) };
    }

    i32 NormalizeScore(i32 score) {
        if (score == 0 || score > ScoreTTWin || score < ScoreTTLoss)
            return score;

        return static_cast<i32>((score * 100) / EvalNormalization);
    }
}
