#include "common.h"

using namespace std;

double w_seq_memset(){
    volatile data_t* vol_a = a;
    double elapsed = get_time();
    #pragma omp parallel
    {
        uint64_t tid = omp_get_thread_num();
        uint64_t elem_per_thread = HS_ARRAY_ELEM / omp_get_num_threads();
        uint64_t offset = tid * elem_per_thread;        
        memset(a + offset, 7, elem_per_thread * sizeof(data_t));
    }
    return get_time() - elapsed;
}

double w_seq_fill(){
    volatile data_t* vol_a = a;
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        vol_a[i] = 7;
    }
    return get_time() - elapsed;
}

double w_rand_ind(){
    volatile data_t * vol_a = a;
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        uint64_t idx = ((i * 0xDEADBEEF) ^ 0xC0FFEE0B) % HS_ARRAY_ELEM;
        vol_a[idx] = 7;
    }
    return get_time() - elapsed;
}

double w_stride_2(){
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 2) {
        a[i] = 7;
    }
    return get_time() - elapsed;
}

double w_stride_4(){
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 4) {
        a[i] = 7;
    }
    return get_time() - elapsed;
}

double w_stride_8(){
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 8) {
        a[i] = 7;
    }
    return get_time() - elapsed;
}

double w_stride_16(){
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 16) {
        a[i] = 7;
    }
    return get_time() - elapsed;
}

double w_stride_32(){
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 32) {
        a[i] = 7;
    }
    return get_time() - elapsed;
}

