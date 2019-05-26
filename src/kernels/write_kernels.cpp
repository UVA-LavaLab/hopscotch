#include "common.h"
#include <string.h>

double kern_seq_write(){
    volatile data_t* vol_a = a;
    double elapsed = mysecond();
    //#pragma omp parallel for
    //for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
    //    vol_a[i] = 7;
    //}
    memset(a, 7, HS_ARRAY_ELEM * sizeof(data_t));
    return mysecond() - elapsed;
}

double kern_stride2_write(){
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 2) {
        a[i] = 7;
    }
    return mysecond() - elapsed;
}

double kern_stride4_write(){
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 4) {
        a[i] = 7;
    }
    return mysecond() - elapsed;
}

double kern_stride8_write(){
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 8) {
        a[i] = 7;
    }
    return mysecond() - elapsed;
}

double kern_stride16_write(){
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 16) {
        a[i] = 7;
    }
    return mysecond() - elapsed;
}

double kern_stride32_write(){
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 32) {
        a[i] = 7;
    }
    return mysecond() - elapsed;
}

