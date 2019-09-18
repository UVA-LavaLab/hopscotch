/*******************************************************************************
 *
 * File: common.cpp
 * Description: Contains definitions of common utility functions.
 * 
 * Author: Alif Ahmed
 * Date: Aug 06, 2019
 *
 ******************************************************************************/
#include "common.h"
#include <cstdlib>
#include <cstdio>
#include <chrono>


/*******************************************************************************
 * Allocates 4096 bytes aligned memory. Portable.
 ******************************************************************************/
void* hs_alloc(size_t size){
    void* ptr = aligned_alloc(4096, size);
    if(ptr == nullptr) {
        fprintf(stderr, "Memory allocation of size %lu bytes failed\n", size);
        exit(-1);
    }
    return ptr;
}



/*******************************************************************************
 * Captures current time as a time_point object. Use hs_duration() to get elapsed
 * time. Portable.
 ******************************************************************************/
std::chrono::high_resolution_clock::time_point hs_get_time() {
    return std::chrono::high_resolution_clock::now();
}



/*******************************************************************************
 * Measures the elasped time (in seconds) since 'start'. Can support upto
 * nanosecond resoultion depending on hardware support.
 ******************************************************************************/
double hs_duration(const std::chrono::high_resolution_clock::time_point &start) {
    std::chrono::duration<double> t = hs_get_time() - start;
    return t.count();
}



/*******************************************************************************
 * Initializes an array with constant value.
 ******************************************************************************/
extern void hs_init_const(DATA_T* arr, uint64_t num_elem, const DATA_T val) {
    #pragma omp parallel for
    for(size_t i = 0; i < num_elem; ++i){
        arr[i] = val;
    }
}

