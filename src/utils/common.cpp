#include <sys/time.h>
#include "common.h"
#include <stdio.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <omp.h>

data_t* a = NULL;
data_t* b = NULL;
data_t* c = NULL;
void** ptr = NULL;

//uint64_t idx1[HS_ARRAY_ELEM];
//uint64_t idx2[HS_ARRAY_ELEM];

void alloc_a(){
    a = (data_t*)malloc(HS_ARRAY_SIZE_BTYE);
}

void alloc_b(){
    b = (data_t*)malloc(HS_ARRAY_SIZE_BTYE);
}

void alloc_ptr(){
    ptr = (void**)malloc(HS_ARRAY_ELEM * sizeof(void*));
}

void free_a(){
    if(a) free(a);
}

void free_b() {
    if(b) free(b);
}

void free_ptr() {
    if(ptr) free(ptr);
}

double get_time() {
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

double kernel_sum_time(uint64_t* iter, double (*func)(), void (*init)()) {
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
}

void no_init(){
    //do nothing
}

void print_thread_num() {
    #pragma omp parallel
    #pragma omp single
    printf("Number of threads = %d\n", omp_get_num_threads());
}

void default_init(){
    memset(a, 1, HS_ARRAY_SIZE_BTYE);
    memset(b, 2, HS_ARRAY_SIZE_BTYE);
    memset(c, 3, HS_ARRAY_SIZE_BTYE);
}

void init_a(){
    memset(a, 1, HS_ARRAY_SIZE_BTYE);
}

void init_ab(){
    memset(a, 1, HS_ARRAY_SIZE_BTYE);
    memset(b, 2, HS_ARRAY_SIZE_BTYE);
}

//init pointer chasing to array 'ptr' with hamiltonian cycle
void init_pointer_chasing() {
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