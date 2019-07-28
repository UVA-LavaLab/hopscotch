#include "common.h"

/*double kernel_min_time_1(uint64_t* iter, double (*func)(data_t*), data_t* arg) {
    init_a();
    func(arg);             //warm up
    double elapsed = 0;
    double min = DBL_MAX;
    uint64_t i;
    for(i = 0; (i < ITER_MIN) || (elapsed < ITER_TIMEOUT_SEC); ++i) {
        double curr = func(arg);
        min = curr < min ? curr : min;
        elapsed += curr;
    }
    *iter = i;
    return min;
}

double rw_seq_inc_1(data_t* __restrict__ a_res){
    data_t* a_aligned = (data_t*) __builtin_assume_aligned (a_res, 4096);
    double elapsed = get_time();
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        a_aligned[i]++;
    }
    return get_time() - elapsed;
}*/

static data_t x[HS_ARRAY_ELEM] __attribute__((aligned(64)));
static data_t y[HS_ARRAY_ELEM] __attribute__((aligned(64)));

int main(){
    /*alloc_a();
    double elapsed;
    uint64_t iter;
    print_thread_num();
    printf("data size: %lu bytes\n", sizeof(data_t));
    print_bw_header();
    
    
    elapsed = kernel_min_time(&iter, rw_seq_inc, init_a);
    print_bw("inc: current", iter, elapsed, HS_ARRAY_SIZE_MB * 2 / elapsed);
    
    elapsed = kernel_min_time_1(&iter, rw_seq_inc_1, a);
    print_bw("inc: restrict", iter, elapsed, HS_ARRAY_SIZE_MB * 2 / elapsed);
    
    free_a();*/
    
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i){
        //x[i] = 1.0;
        y[i] = 2.0;
    }
    
    double elapsed = get_time();
    
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i){
        x[i] = y[i];
    }
    
    elapsed = get_time() - elapsed;
    printf("%0.0lf\n", HS_ARRAY_SIZE_MB * 2 / elapsed);

    data_t sum = 0;
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i){
        sum += x[i];
    }
    printf("%0.0lf\n", sum);
    
    return 0;
}