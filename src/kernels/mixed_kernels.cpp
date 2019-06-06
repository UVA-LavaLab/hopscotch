#include "common.h"
#include <string.h>

double rw_seq_memcpy(){
    double elapsed = get_time();
    #pragma omp parallel
    {
        uint64_t tid = omp_get_thread_num();
        uint64_t elem_per_thread = HS_ARRAY_ELEM / omp_get_num_threads();
        uint64_t offset = tid * elem_per_thread;        
        memcpy(a + offset, b + offset, elem_per_thread * sizeof(data_t));
    }
    return get_time() - elapsed;
}

double rw_seq_copy(){
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        a[i] = b[i];
    }
    return get_time() - elapsed;
}

double rw_seq_rmw(){
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        a[i]++;
    }
    return get_time() - elapsed;
}

