#include "common.h"
#include "stdio.h"
#include <immintrin.h>
#include "iacaMarks.h"

#define FLOP_PER_ELEM       512
#define NUM_REP     (FLOP_PER_ELEM/2)

#define KERN1(p,q)          p = p * p + q;
#define KERN2(p,q)          KERN1(p,q); KERN1(p,q);
#define KERN4(p,q)          KERN2(p,q); KERN2(p,q);
#define KERN8(p,q)          KERN4(p,q); KERN4(p,q);
#define KERN16(p,q)          KERN8(p,q); KERN8(p,q);
#define KERN32(p,q)      KERN16(p,q); KERN16(p,q);
#define KERN64(p,q)      KERN32(p,q); KERN32(p,q);
#define KERN128(p,q)     KERN64(p,q); KERN64(p,q);
#define KERN256(p,q)     KERN128(p,q); KERN128(p,q);
#define KERN512(p,q)     KERN256(p,q); KERN256(p,q);
#define KERN1024(p,q)    KERN512(p,q); KERN512(p,q);
#define KERN2048(p,q)    KERN1024(p,q); KERN1024(p,q);
#define KERN4096(p,q)    KERN2048(p,q); KERN2048(p,q);

#define KERN_REP_NX(x,p,q)   KERN ## x (p,q)
#define KERN_REP(x,p,q)     KERN_REP_NX(x,p,q)

double kernel_roofline(data_t* x, data_t r, data_t s) {
    data_t* __restrict__ x_al = (data_t*)__builtin_assume_aligned(x, 4096);
    const __m256d rv = _mm256_set1_pd(r);
    const __m256d sv = _mm256_set1_pd(s);
    double elapsed = get_time();
    #pragma omp parallel for
    for(uint64_t i = 0; i < HS_ARRAY_ELEM; i+=32){
        IACA_START
        __m256d xv1 = _mm256_load_pd(&x[i]);
        __m256d xv2 = _mm256_load_pd(&x[i+4]);
        __m256d xv3 = _mm256_load_pd(&x[i+8]);
        __m256d xv4 = _mm256_load_pd(&x[i+12]);
        __m256d xv5 = _mm256_load_pd(&x[i+16]);
        __m256d xv6 = _mm256_load_pd(&x[i+20]);
        __m256d xv7 = _mm256_load_pd(&x[i+24]);
        __m256d xv8 = _mm256_load_pd(&x[i+28]);
        
        for(uint64_t j = 0; j < FLOP_PER_ELEM/2; ++j) {
            xv1 = _mm256_fmadd_pd(xv1, rv, sv);
            xv2 = _mm256_fmadd_pd(xv2, rv, sv);
            xv3 = _mm256_fmadd_pd(xv3, rv, sv);
            xv4 = _mm256_fmadd_pd(xv4, rv, sv);
            xv5 = _mm256_fmadd_pd(xv5, rv, sv);
            xv6 = _mm256_fmadd_pd(xv6, rv, sv);
            xv7 = _mm256_fmadd_pd(xv7, rv, sv);
            xv8 = _mm256_fmadd_pd(xv8, rv, sv);
        }
               
        _mm256_store_pd(&x[i], xv1);
        _mm256_store_pd(&x[i+4], xv2);
        _mm256_store_pd(&x[i+8], xv3);
        _mm256_store_pd(&x[i+12], xv4);
        _mm256_store_pd(&x[i+16], xv5);
        _mm256_store_pd(&x[i+20], xv6);
        _mm256_store_pd(&x[i+24], xv7);
        _mm256_store_pd(&x[i+28], xv8);
    }
    IACA_END
    return get_time() - elapsed;
}

#define NUM_ITER    200

int main(){
    alloc_a();
    init_a();
    
    double ai = FLOP_PER_ELEM / 16.0;

    double curr_time;
    double min_time = DBL_MAX;
    for(uint64_t i = 0; i < NUM_ITER; ++i){
        curr_time = kernel_roofline(a, 1.0 + 1e-8, 1e-8);
        if(curr_time < min_time){
            min_time = curr_time;
        }
        printf("AI: %lf    FLOPS: %lf\n", ai, (HS_ARRAY_SIZE_MB * 2 / curr_time) * ai);
    }
    
    printf("BEST: AI: %lf    FLOPS: %lf\n", ai, (HS_ARRAY_SIZE_MB * 2/ min_time) * ai);
    free_a();

    return 0;
}
