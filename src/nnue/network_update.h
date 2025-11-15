#pragma once

#include "../defs.h"

namespace Horsie {

    struct PerspectiveUpdate {
        std::array<i32, 2> Adds{};
        std::array<i32, 2> Subs{};
        i32 AddCnt{};
        i32 SubCnt{};

        void Clear() {
            AddCnt = SubCnt = 0;
        }

        void PushSub(i32 sub1) {
            Subs[SubCnt++] = sub1;
        }

        void PushSubAdd(i32 sub1, i32 add1) {
            Subs[SubCnt++] = sub1;
            Adds[AddCnt++] = add1;
        }

        void PushSubSubAdd(i32 sub1, i32 sub2, i32 add1) {
            Subs[SubCnt++] = sub1;
            Subs[SubCnt++] = sub2;
            Adds[AddCnt++] = add1;
        }

        void PushSubSubAddAdd(i32 sub1, i32 sub2, i32 add1, i32 add2) {
            Subs[SubCnt++] = sub1;
            Subs[SubCnt++] = sub2;
            Adds[AddCnt++] = add1;
            Adds[AddCnt++] = add2;
        }
    };

    struct NetworkUpdate {
        std::array<PerspectiveUpdate, 2> Perspectives;

        PerspectiveUpdate& operator[](const i32 c) { return Perspectives[c]; }
    };

}
