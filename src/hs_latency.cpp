#include "common.h"

int main(){
    double elapsed;
    uint64_t iter;
    alloc_ptr();
    
    elapsed = kernel_sum_time(&iter, r_rand_pchase, init_pointer_chasing);
#if HS_ARRAY_ELEM < ELEM_MIN
    uint64_t iter_count = ELEM_MIN / HS_ARRAY_ELEM;
    double latency = elapsed * 1e9 / iter / HS_ARRAY_ELEM / iter_count;
#else
    double latency = elapsed * 1e9 / iter / HS_ARRAY_ELEM;
#endif
    
    printf("latency: %lf (ns/access)\n", latency);
        
    free_ptr();
    
    return 0;
}