/*******************************************************************************
 *
 * File: common.h
 * Description: Common header file.
 * 
 * Author: Alif Ahmed
 * Date: Aug 06, 2019
 *
 ******************************************************************************/
#ifndef __HS_COMMON_H__
#define __HS_COMMON_H__

#define XSTR(s) STR(s)
#define STR(s)  #s

extern void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#endif
