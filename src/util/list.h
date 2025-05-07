#pragma once

#include "../defs.h"

#include <algorithm>
#include <array>
#include <cassert>

namespace Horsie::Util {
    template <class T, size_t Capacity>
    class List {
    public:
        List() : Count(0) { }

        constexpr auto Size() const { return Count; }
        constexpr bool Empty() const { return Count == 0; }
        constexpr void Clear() { Count = 0; }

        constexpr T& operator[](int i) {
            assert(i >= 0 && i < Count);
            return Items[i];
        }

        constexpr T operator[](int i) const {
            assert(i >= 0 && i < Count);
            return Items[i];
        }

        constexpr void Add(const T& object) {
            assert(Count < Capacity);
            Items[Count++] = object;
        }

        constexpr T& RemoveLast() {
            assert(Count > 0);
            return Items[--Count];
        }

        constexpr T& Last() { return Items[Count - 1]; }

        constexpr auto& Raw() { return Items; }

    private:
        std::array<T, Capacity> Items;
        i32 Count = 0;
    };
}
