#include "common.h"
#include <string.h>

double rw_seq_copy(){
    double elapsed = get_time();
    #pragma omp parallel for simd aligned(a,b:64)
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        a[i] = b[i];
    }
    return get_time() - elapsed;
}

double rw_seq_inc(){
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        a[i]++;
    }
    return get_time() - elapsed;
}

double rw_seq_scan(){
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 1; i < HS_ARRAY_ELEM; ++i) {
        a[i] += a[i-1];
    }
    return get_time() - elapsed;
}

double rw_gather(){
    double elapsed = get_time();
    #pragma omp parallel for simd
    for(uint64_t i = 1; i < HS_ARRAY_ELEM; ++i) {
        a[i] = b[idx1[i]];
    }
    return get_time() - elapsed;
}

double rw_scatter(){
    double elapsed = get_time();
    #pragma omp parallel for simd aligned (a,b,idx1,idx2:64)
    for(uint64_t i = 1; i < HS_ARRAY_ELEM; ++i) {
        a[idx1[i]] = b[i];
    }
    return get_time() - elapsed;
}

double rw_scatter_gather(){
    double elapsed = get_time();
    #pragma omp parallel for simd aligned (a,b,idx1,idx2:64)
    for(uint64_t i = 1; i < HS_ARRAY_ELEM; ++i) {
        a[idx1[i]] = b[idx2[i]];
    }
    return get_time() - elapsed;
}

double rw_tile(uint64_t L, uint64_t K){
    double elapsed = get_time();
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += K) {
        for(uint64_t j = 0; j < L; j++) {
            const uint64_t idx = i + j;
            if(idx >= HS_ARRAY_ELEM)
                break;
            a[idx]++;
        }
    }
    return get_time() - elapsed;
}
