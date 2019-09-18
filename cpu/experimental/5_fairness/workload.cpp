/*******************************************************************************
 *
 * File: workload.cpp
 * Description: TODO
 * 
 * Author: Alif Ahmed
 * Date: Aug 06, 2019
 *
 ******************************************************************************/
#include "common.h"
#include <cstdint>
#include <iostream>
#include <cfloat>

using namespace std;

#undef  NTRIES
#define NTRIES          10UL

inline void seq_kernel(){
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i){
        a[i]++;
    }
}

inline void rand_kernel(){
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        uint64_t idx = ((i * 0xDEADBEEF) ^ 0xC0FFEE0B) % HS_ARRAY_ELEM;
        a[idx]++;
    }
}

int main(){
    double tseq = 0;
    double trand = 0;
    uint64_t iter;

    alloc_a();
    init_a();

    seq_kernel();       //warm up
    for(uint64_t n = 0; n < NTRIES; ++n){ 
        double t = get_time();
        seq_kernel();
        t = get_time() - t;
        tseq += t;
    }   
    cout << "sequential: " << HS_ARRAY_SIZE_MB * 2 * NTRIES / tseq << " MB/s" << endl;

    rand_kernel();      //warm up
    for(uint64_t n = 0; n < NTRIES; ++n) {
        double t = get_time();
        rand_kernel();
        t = get_time() - t;
        trand += t;
    }
    cout << "random: " << HS_ARRAY_SIZE_MB * 2 * NTRIES / trand << " MB/s" << endl;

    // Prevents elimination of previous block during optimization.
    uint64_t sum = 0;
    #pragma omp parallel for reduction (+:sum)
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        sum += a[i];
    }

    cout << sum << endl;

    free_a();
    return 0;
}
