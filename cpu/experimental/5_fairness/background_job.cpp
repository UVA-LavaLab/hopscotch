/*******************************************************************************
 *
 * File: background_job.cpp
 * Description: Runs a sequential access kernel for infinite time. Number of
 *              threads is configurable.
 * 
 * Author: Alif Ahmed
 * Date: Aug 06, 2019
 *
 ******************************************************************************/
#include "common.h"
#include <cstdint>
#include <iostream>

using namespace std;

#undef  NTRIES
#define NTRIES          100000UL

int main(int argc, const char** argv){
    if(argc != 2){
        cout << "usage: ./back <num_threads>" << endl;
        exit(-1);
    }
    omp_set_num_threads(atoi(argv[1]));
    
    #pragma omp parallel
    #pragma omp single
    cout << "Number of threads: " << omp_get_num_threads() << endl;

    alloc_a();
    init_a();

    for(uint64_t n = 0; n < NTRIES; ++n){
        #pragma omp parallel for simd
        for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i){
            a[i]++;
        }
    }

    // Set NTRIES large enough so that control does not reach this point.
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
