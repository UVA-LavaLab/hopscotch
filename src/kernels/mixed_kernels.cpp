#include "common.h"
#include <string.h>

double rw_seq_memcpy(){
    double elapsed = mysecond();
    memcpy(a, b, HS_ARRAY_SIZE_BTYE);
    return mysecond() - elapsed;
}

double rw_seq_copy(){
    double elapsed = mysecond();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; ++i) {
        a[i] = b[i];
    }
    return mysecond() - elapsed;
}

