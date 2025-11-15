#pragma once

#include "../defs.h"

namespace Horsie {
    void DbgHit(bool cond, i32 slot = 0);
    void DbgPrint();
    void DbgMean(i64 value, i32 slot);
    void DbgMinMax(i64 value, i32 slot);
}
