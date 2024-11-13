#pragma once


#ifndef NETWORK_UPDATE_H
#define NETWORK_UPDATE_H


namespace Horsie {

	struct PerspectiveUpdate {
        int Adds[2];
        int Subs[2];
        int AddCnt = 0;
        int SubCnt = 0;

        void Clear()
        {
            AddCnt = SubCnt = 0;
        }

        void PushSub(int sub1)
        {
            Subs[SubCnt++] = sub1;
        }

        void PushSubAdd(int sub1, int add1)
        {
            Subs[SubCnt++] = sub1;
            Adds[AddCnt++] = add1;
        }

        void PushSubSubAdd(int sub1, int sub2, int add1)
        {
            Subs[SubCnt++] = sub1;
            Subs[SubCnt++] = sub2;
            Adds[AddCnt++] = add1;
        }

        void PushSubSubAddAdd(int sub1, int sub2, int add1, int add2)
        {
            Subs[SubCnt++] = sub1;
            Subs[SubCnt++] = sub2;
            Adds[AddCnt++] = add1;
            Adds[AddCnt++] = add2;
        }
	};

    struct NetworkUpdate {
        PerspectiveUpdate Perspectives[2];

        PerspectiveUpdate& operator[](const int c) { return Perspectives[c]; }
    };

}

#endif