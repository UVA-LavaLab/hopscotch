#include "common.h"
#include <string.h>

double rw_seq_copy(){
    double elapsed = get_time();
    #pragma omp parallel for
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


