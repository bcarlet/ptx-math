#ifndef __CUDA_CHECK_H
#define __CUDA_CHECK_H

#include <cuda_runtime.h>

#define CUDA_CHECK(errcode) { cuda_err_check((errcode), __FILE__, __LINE__); }

void cuda_err_check(cudaError_t code, const char *file, int line);

#endif
