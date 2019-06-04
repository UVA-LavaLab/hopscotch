#include "common.h"

double r_seq_ind(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        uint64_t res = vol_a[i];
    }
    return mysecond() - elapsed;
}

double r_seq_reduce(){
    volatile data_t * vol_a = a;
    volatile uint64_t sum = 0;
    double elapsed = mysecond();
    #pragma omp parallel for reduction(+ : sum)
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        sum += vol_a[i];
    }
    return mysecond() - elapsed;
}

double r_rand_pchase(){
    void **curr = ptr;
    void* saved_val = *curr;
    double elapsed = mysecond();
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        curr = (void**)*curr;
    }
    elapsed = mysecond() - elapsed;
    *curr = saved_val; //do a write to prevent optimization
    return elapsed;
}

double r_rand_ind(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        uint64_t idx = ((i * 0xDEADBEEF) ^ 0xC0FFEE0B) % HS_ARRAY_ELEM;
        volatile uint64_t res = vol_a[idx];
    }
    return mysecond() - elapsed;
}

double r_stride_2(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 2) {
        volatile register uint64_t res = vol_a[i];
    }
    return mysecond() - elapsed;
}

double r_stride_4(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 4) {
        volatile register uint64_t res = vol_a[i];
    }
    return mysecond() - elapsed;
}

double r_stride_8(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 8) {
        volatile register uint64_t res = vol_a[i];
    }
    return mysecond() - elapsed;
}

double r_stride_16(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 16) {
        volatile register uint64_t res = vol_a[i];
    }
    return mysecond() - elapsed;
}

double r_stride_32(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 32) {
        volatile register uint64_t res = vol_a[i];
    }
    return mysecond() - elapsed;
}

