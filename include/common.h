#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t data_t;

#ifndef HS_ARRAY_ELEM
#define HS_ARRAY_ELEM       (32*1024*1024UL)
#endif

#define ITER_MIN            8
#define ITER_TIMEOUT_SEC    2

#define HS_ARRAY_SIZE_BTYE  (sizeof(data_t)*HS_ARRAY_ELEM)
#define HS_ARRAY_SIZE_MB    (HS_ARRAY_SIZE_BTYE/1024/1024)

// Globals
extern data_t a[HS_ARRAY_ELEM];
extern data_t b[HS_ARRAY_ELEM];
extern data_t c[HS_ARRAY_ELEM];
extern void* ptr[HS_ARRAY_ELEM];

//extern uint64_t idx1[HS_ARRAY_ELEM];
//extern uint64_t idx2[HS_ARRAY_ELEM];

//helper functions for print formatting
#define print_bw_header()               printf("\n%-30s%-15s%-15s%-15s\n", "Kernel", "Iterations", "Time (s)", "BW (MB/s)");
#define print_bw(kernel,iter,time,bw)   printf("%-30s%-15lu%-15.2lf%-15.0lf\n",kernel,iter,time,bw);


//Forward declarations
extern double mysecond();
extern void default_init();
extern void init_a();
extern void init_ab();
extern void no_init();
extern void init_pointer_chasing();
extern double kernel_sum_time(uint64_t* iter, double (*func)(), void (*init)());
extern double kernel_min_time(uint64_t* iter, double (*func)(), void (*init)());

//read kernels
extern double r_seq_ind();
extern double r_seq_reduce();
extern double r_rand_ind();
extern double r_rand_pchase();
extern double r_stride_2();
extern double r_stride_4();
extern double r_stride_8();
extern double r_stride_16();
extern double r_stride_32();

//write kernels
extern double w_seq_memset();
extern double w_seq_fill();
extern double w_stride_2();
extern double w_stride_4();
extern double w_stride_8();
extern double w_stride_16();
extern double w_stride_32();

//mixed kernels
extern double rw_seq_memcpy();
extern double rw_seq_copy();

#endif /* COMMON_H */

