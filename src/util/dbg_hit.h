#pragma once

#ifndef DBG_HIT_H
#define DBG_HIT_H 1

#include "../types.h"

namespace Horsie {
    void dbg_hit_on(bool cond, int slot = 0);
    void dbg_mean_of(int64_t value, int slot = 0);
    void dbg_stdev_of(int64_t value, int slot = 0);
    void dbg_extremes_of(int64_t value, int slot = 0);
    void dbg_correl_of(int64_t value1, int64_t value2, int slot = 0);
    void dbg_print();
}


#endif // !DBG_HIT_H