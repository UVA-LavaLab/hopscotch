#include <sys/time.h>
#include "common.h"
#include <vector>
#include <algorithm>

data_t* __restrict__ a = NULL;
data_t* __restrict__ b = NULL;
data_t* __restrict__ c = NULL;
void** ptr = NULL;
uint64_t* __restrict__ idx1 = NULL;
uint64_t* __restrict__ idx2 = NULL;

void* alloc_aligned_4K(uint64_t size){
    void* ptr;
    if(posix_memalign(&ptr, 4096, size)){
        fprintf(stderr, "Memory allocation of size %luB failed\n", size);
        exit(-1);
    }
    return ptr;
}

void alloc_a(){
    //a = (data_t*)malloc(HS_ARRAY_SIZE_BTYE);
    a = (data_t*)alloc_aligned_4K(HS_ARRAY_SIZE_BTYE);
}

void alloc_b(){
    //b = (data_t*)malloc(HS_ARRAY_SIZE_BTYE);
    b = (data_t*)alloc_aligned_4K(HS_ARRAY_SIZE_BTYE);
}

void alloc_ptr(){
    //ptr = (void**)malloc(HS_ARRAY_ELEM * sizeof(void*));
    ptr = (void**)alloc_aligned_4K(HS_ARRAY_ELEM * sizeof(void*));
}

void alloc_idx1(){
    //idx1 = (uint64_t*)malloc(HS_ARRAY_ELEM * sizeof(uint64_t));
    idx1 = (uint64_t*)alloc_aligned_4K(HS_ARRAY_ELEM * sizeof(uint64_t));
}

void alloc_idx2(){
    //idx2 = (uint64_t*)malloc(HS_ARRAY_ELEM * sizeof(uint64_t));
    idx2 = (uint64_t*)alloc_aligned_4K(HS_ARRAY_ELEM * sizeof(uint64_t));
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

void free_idx1(){
    if(idx1) free(idx1);
}

void free_idx2(){
    if(idx2) free(idx2);
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
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i){
        a[i] = 1.0 + 1e-8;
    }
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
}
