#include "common.h"
#include <string.h>

double w_seq_memset(){
    volatile data_t* vol_a = a;
    double elapsed = mysecond();
    memset(a, 7, HS_ARRAY_ELEM * sizeof(data_t));
    return mysecond() - elapsed;
}

double w_seq_fill(){
    volatile data_t* vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        vol_a[i] = 7;
    }
    return mysecond() - elapsed;
}

double w_stride_2(){
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 2) {
        a[i] = 7;
    }
    return mysecond() - elapsed;
}

double w_stride_4(){
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 4) {
        a[i] = 7;
    }
    return mysecond() - elapsed;
}

double w_stride_8(){
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 8) {
        a[i] = 7;
    }
    return mysecond() - elapsed;
}

double w_stride_16(){
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 16) {
        a[i] = 7;
    }
    return mysecond() - elapsed;
}

double w_stride_32(){
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 32) {
        a[i] = 7;
    }
    return mysecond() - elapsed;
}

