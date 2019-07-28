#include "common.h"

double r_seq_ind(){
    volatile data_t * vol_a = a;
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        volatile data_t res = vol_a[i];
    }
    return get_time() - elapsed;
}

double r_seq_reduce(){
    volatile data_t * vol_a = a;
    volatile data_t sum = 0;
    double elapsed = get_time();
    #pragma omp parallel for simd reduction(+ : sum)
    //#pragma omp simd reduction(+ : sum)
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        sum += a[i];
    }
    return get_time() - elapsed;
}

double r_rand_pchase(){
    void **curr = ptr;
    void* saved_val = *ptr;
    double elapsed = get_time();
#if HS_ARRAY_ELEM < ELEM_MIN
    for(uint64_t k = ELEM_MIN / HS_ARRAY_ELEM; k > 0 ; --k) {
#endif
    for(uint64_t i = HS_ARRAY_ELEM; i > 0; i -= 8) {
        //manual loop unroll
        curr = (void**)*curr;
        curr = (void**)*curr;
        curr = (void**)*curr;
        curr = (void**)*curr;
        curr = (void**)*curr;
        curr = (void**)*curr;
        curr = (void**)*curr;
        curr = (void**)*curr;
    }
#if HS_ARRAY_ELEM < ELEM_MIN
    }
#endif
    elapsed = get_time() - elapsed;
    *curr = saved_val; //do a write to prevent optimization
    return elapsed;
}

double r_rand_ind(){
    volatile data_t * vol_a = a;
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        uint64_t idx = ((i * 0xDEADBEEF) ^ 0xC0FFEE0B) % HS_ARRAY_ELEM;
        volatile data_t res = vol_a[idx];
    }
    return get_time() - elapsed;
}

double r_stride_2(){
    volatile data_t * vol_a = a;
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 2) {
        volatile register data_t res = vol_a[i];
    }
    return get_time() - elapsed;
}

double r_stride_4(){
    volatile data_t * vol_a = a;
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 4) {
        volatile register data_t res = vol_a[i];
    }
    return get_time() - elapsed;
}

double r_stride_8(){
    volatile data_t * vol_a = a;
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 8) {
        volatile register data_t res = vol_a[i];
    }
    return get_time() - elapsed;
}

double r_stride_16(){
    volatile data_t * vol_a = a;
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 16) {
        volatile register data_t res = vol_a[i];
    }
    return get_time() - elapsed;
}

double r_stride_32(){
    volatile data_t * vol_a = a;
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 32) {
        volatile register data_t res = vol_a[i];
    }
    return get_time() - elapsed;
}

double r_tile(uint64_t L, uint64_t K){
//double r_tile(){
    volatile data_t * vol_a = a;
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += K) {
        for(uint64_t j = 0; j < L; j++) {
            volatile data_t res = vol_a[i+j];
        }
    }
    return get_time() - elapsed;
}

double r_dma(data_t** addr, uint64_t* len, uint64_t count) {
    volatile data_t * vol_a = a;
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < count; i++) {
        const uint64_t c_len = len[i];
        volatile const data_t* c_addr = addr[i];
        for(uint64_t j = 0; j < c_len; j++) {
            volatile data_t res = *c_addr++;
        }
    }
    return get_time() - elapsed;
}