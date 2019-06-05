#include "common.h"
#include <string.h>

double rw_seq_memcpy(){
    double elapsed = get_time();
    memcpy(a, b, HS_ARRAY_SIZE_BTYE);
    return get_time() - elapsed;
}

double rw_seq_copy(){
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        a[i] = b[i];
    }
    return get_time() - elapsed;
}

