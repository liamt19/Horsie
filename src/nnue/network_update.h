#pragma once

#ifndef NETWORK_UPDATE_H
#define NETWORK_UPDATE_H

namespace Horsie {

    struct PerspectiveUpdate {
        i32 Adds[2];
        i32 Subs[2];
        i32 AddCnt = 0;
        i32 SubCnt = 0;

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
        PerspectiveUpdate Perspectives[2];

        PerspectiveUpdate& operator[](const i32 c) { return Perspectives[c]; }
    };

}


#endif // !NETWORK_UPDATE_H