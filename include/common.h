#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef uint64_t data_t;

#ifndef HS_ARRAY_ELEM
#define HS_ARRAY_ELEM       (32*1024*1024UL)
#endif

#define ITER_MIN            8
#define ITER_TIMEOUT_SEC    10

#define HS_ARRAY_SIZE_MB    (sizeof(data_t)*HS_ARRAY_ELEM/1024/1024)

// Globals
extern data_t a[HS_ARRAY_ELEM];
extern data_t b[HS_ARRAY_ELEM];
extern data_t c[HS_ARRAY_ELEM];
extern void* ptr[HS_ARRAY_ELEM];

//extern uint64_t idx1[HS_ARRAY_ELEM];
//extern uint64_t idx2[HS_ARRAY_ELEM];

//Forward declarations
extern double mysecond();
extern void default_init();
extern void no_init();
extern void init_pointer_chasing();
extern double run_kernel(uint64_t* iter, double (*func)(), void (*init)());

//read kernels
extern double kern_seq_read();
extern double kern_seq_reduction();
extern double kern_rand_read();
extern double kern_read_pointer_chasing();
extern double kern_stride2_read();
extern double kern_stride4_read();
extern double kern_stride8_read();
extern double kern_stride16_read();
extern double kern_stride32_read();

//write kernels
extern double kern_seq_write();

#endif /* COMMON_H */

