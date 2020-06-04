#ifndef CUDA_UTIL_HPP
#define CUDA_UTIL_HPP

#include <cuda_runtime.h>

/**
 * Sets up a grid-stride loop for use in a CUDA kernel.
 */
#define GRID_STRIDE_LOOP(indexName, n) \
    for (int indexName = blockIdx.x * blockDim.x + threadIdx.x; \
         indexName < (n); \
         indexName += blockDim.x * gridDim.x)

/**
 * Convenience macro for CUDA error checking.
 */
#define CUDA_CHECK(errcode) cuda_error_check((errcode), __FILE__, __LINE__)

/**
 * Print the available compute-capable devices.
 */
void print_devices();

/**
 * If code is not cudaSuccess, print an error message and terminate the program.
 */
void cuda_error_check(cudaError_t code, const char *file, int line);

#endif
