#include <sys/time.h>
#include "common.h"
#include <vector>
#include <algorithm>

data_t a[HS_ARRAY_ELEM];
data_t b[HS_ARRAY_ELEM];
data_t c[HS_ARRAY_ELEM];
void* ptr[HS_ARRAY_ELEM];

uint64_t idx1[HS_ARRAY_ELEM];
uint64_t idx2[HS_ARRAY_ELEM];

double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp,&tzp);
	return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

double run_kernel(uint64_t* iter, double (*func)(), void (*init)()) {
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

void no_init(){
    //do nothing
}

void default_init(){
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        a[i] = 1;
        //b[i] = 2;
        //c[i] = 3;
    }
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