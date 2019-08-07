#include "common.h"
#include <cstdlib>
#include <cstdio>
#include <chrono>

void* hs_alloc(size_t size){
    void* ptr = aligned_alloc(4096, size);
	if(ptr == nullptr) {
		fprintf(stderr, "Memory allocation of size %lu bytes failed\n", size);
        exit(-1);
	}
    return ptr;
}

std::chrono::high_resolution_clock::time_point hs_get_time() {
	return std::chrono::high_resolution_clock::now();
}

double hs_duration(const std::chrono::high_resolution_clock::time_point &start) {
	std::chrono::duration<double> t = hs_get_time() - start;
	return t.count();
}

extern void hs_init_const(DATA_T* arr, uint64_t num_elem, const DATA_T val) {
	#pragma omp parallel for
	for(size_t i = 0; i < num_elem; ++i){
		arr[i] = val;
	}
}

/*double kernel_sum_time(uint64_t* iter, double (*func)(), void (*init)()) {
    init();
    func();             //warm up
    double elapsed = 0;
    uint64_t i;
    for(i = 0; (i < ITER_MIN) || (elapsed < ITER_TIMEOUT_SEC); ++i) {
        elapsed += func();
    }
    *iter = i;
    return elapsed;
}

double kernel_min_time(uint64_t* iter, double (*func)(), void (*init)()) {
    init();
    func();             //warm up
    double elapsed = 0;
    double min = DBL_MAX;
    uint64_t i;
    for(i = 0; (i < ITER_MIN) || (elapsed < ITER_TIMEOUT_SEC); ++i) {
        double curr = func();
        min = curr < min ? curr : min;
        elapsed += curr;
    }
    *iter = i;
    return min;
}*/

//init pointer chasing to array 'ptr' with hamiltonian cycle
/*void init_pointer_chasing() {
    std::vector<void*> unused(HS_ARRAY_ELEM-1);
    #pragma omp parallel for
    for(uint64_t i = 1; i < HS_ARRAY_ELEM; ++i) {
        unused[i-1] = ptr + i;
    }
    std::random_shuffle(unused.begin(), unused.end());
    void** curr = ptr;
    for(uint64_t i = 0; i < HS_ARRAY_ELEM-1; ++i) {
        *curr = unused[i];
        curr = (void**)unused[i];
    }
    *curr = ptr;
}

void init_idx1(){
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i){
        idx1[i] = i;
    }
    std::random_shuffle(idx1, idx1 + HS_ARRAY_ELEM);
}

void init_idx12(){
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i){
        idx1[i] = i;
        idx2[i] = i;
    }
    std::random_shuffle(idx1, idx1 + HS_ARRAY_ELEM);
    std::random_shuffle(idx2, idx2 + HS_ARRAY_ELEM);
}*/
