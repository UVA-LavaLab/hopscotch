#include "common.h"

double kern_seq_read(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        uint64_t res = vol_a[i];
    }
    return mysecond() - elapsed;
}

double kern_seq_reduction(){
    volatile data_t * vol_a = a;
    volatile uint64_t sum = 0;
    double elapsed = mysecond();
    #pragma omp parallel for reduction(+ : sum)
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        sum += vol_a[i];
    }
    return mysecond() - elapsed;
}

double kern_read_pointer_chasing(){
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

double kern_rand_read(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        uint64_t idx = ((i * 0xDEADBEEF) ^ 0xC0FFEE0B) % HS_ARRAY_ELEM;
        volatile uint64_t res = vol_a[idx];
    }
    return mysecond() - elapsed;
}

double kern_stride2_read(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 2) {
        volatile register uint64_t res = vol_a[i];
    }
    return mysecond() - elapsed;
}

double kern_stride4_read(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 4) {
        volatile register uint64_t res = vol_a[i];
    }
    return mysecond() - elapsed;
}

double kern_stride8_read(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 8) {
        volatile register uint64_t res = vol_a[i];
    }
    return mysecond() - elapsed;
}

double kern_stride16_read(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 16) {
        volatile register uint64_t res = vol_a[i];
    }
    return mysecond() - elapsed;
}

double kern_stride32_read(){
    volatile data_t * vol_a = a;
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i += 32) {
        volatile register uint64_t res = vol_a[i];
    }
    return mysecond() - elapsed;
}

