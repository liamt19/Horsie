#pragma once

#ifndef ALLOC_H
#define ALLOC_H

#include "../types.h"


namespace Horsie
{
    template <typename T>
    inline auto AlignedAlloc(nuint items, nuint alignment = AllocAlignment) {
        nuint bytes = ((nuint)sizeof(T) * (nuint)items);

#if defined(_MSC_VER) || defined(_WIN32)
        return static_cast<T*>(_aligned_malloc(bytes, alignment));
#else
        return static_cast<T*>(std::aligned_alloc(alignment, bytes));
#endif
    }

    inline auto AlignedFree(void* ptr) {
        if (!ptr)
            return;

#if defined(_MSC_VER) || defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
}


#endif // !ALLOC_H