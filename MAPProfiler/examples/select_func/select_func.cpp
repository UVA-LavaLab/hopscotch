/*******************************************************************************
 *
 * File: select_func.cpp
 * Description: Example for testing function selection.
 * 
 * Author:  Alif Ahmed
 * Email:   alifahmed@virginia.edu
 * Updated: Aug 06, 2019
 *
 ******************************************************************************/
#include <stdlib.h>

#define N   8192

static volatile int arr[N];
static volatile int idx[N];

class AClass{
    public:
    //"AClass::func1" as function name to profile this
     static void func1(){
        //sequential read (idx) + random write (arr)
        for(int i = 0; i < N; ++i) {
            arr[idx[i]] = 2;
        }
    }
};

//"func1" as function name to profile his
void func1(){
    //sequential read (idx) + random read (arr)
    int sum = 0;
    for(int i = 0; i < N; ++i) {
        sum += arr[idx[i]];
    }
}

//"rmw" as function name to profile this
void rmw(){
    //sequential read (idx) + random read-modify-write (arr)
    for(int i=0; i < N; ++i){
        arr[idx[i]]++;
    }
}

//"main" as function name will profile this
//"func" as function name will profile both "AClass::func1" and "func1"
int main(int argc, const char** argv) {
    //create index (sequential write)
    for(int i = 0; i < N; ++i) {
        idx[i] = rand() % N;
    }
    AClass::func1();
    func1();
    rmw();

    return 0;
}
