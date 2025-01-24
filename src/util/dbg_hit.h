#pragma once

#ifndef DBG_HIT_H
#define DBG_HIT_H 1

#include "../defs.h"

namespace Horsie {
    void dbg_hit_on(bool cond, i32 slot = 0);
    void dbg_print();
}


#endif // !DBG_HIT_H