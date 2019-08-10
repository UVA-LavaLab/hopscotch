/*******************************************************************************
 *
 * File: common.h
 * Description: Common header.
 * 
 * Author: Alif Ahmed
 * Date: Aug 06, 2019
 *
 ******************************************************************************/
#ifndef _HS_COMMON_H_
#define _HS_COMMON_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>



/******************************************************************************
 * Defines
 ******************************************************************************/
#define XSTR(s)	STR(s)
#define STR(s)	#s

// Data type encoding. Used to dynamically generate SIMD intrinsics.
#define DATA_TYPE_DOUBLE	0
#define	DATA_TYPE_FLOAT		1

#ifndef DATA_T_ENC
#define DATA_T_ENC			DATA_TYPE_DOUBLE
#endif

#if DATA_T_ENC == DATA_TYPE_DOUBLE
#define DATA_T double
#elif DATA_T_ENC == DATA_TYPE_FLOAT
#define DATA_T float
#else
#error "Unknown data type"
#endif

// Array size. Set by roofline.py during compilation.
#ifndef HS_ARRAY_SIZE_MB
#define HS_ARRAY_SIZE_MB	(1024UL)
#endif

// Number of tries. Best performance is taken.
#ifndef NTRIES
#define NTRIES				5
#endif

// Array size. Derived from HS_ARRAY_SIZE_MB
#define HS_ARRAY_SIZE_KB	(HS_ARRAY_SIZE_MB * 1024UL)
#define HS_ARRAY_SIZE_BYTE	(HS_ARRAY_SIZE_KB * 1024UL)

// Number of array elements in working set
#define HS_ARRAY_ELEM		(HS_ARRAY_SIZE_BYTE/sizeof(DATA_T))


/******************************************************************************
 * Functions
 ******************************************************************************/
//Forward declarations
extern std::chrono::high_resolution_clock::time_point hs_get_time();
extern double hs_duration(const std::chrono::high_resolution_clock::time_point &start);

// Allocates 4K aligned memory. Portable.
extern void* hs_alloc(size_t size);

// Initializes an array with constant value
extern void hs_init_const(DATA_T* arr, uint64_t num_elem, const DATA_T val);

#endif /* COMMON_H */

