#ifndef UTIL_CUDA_CUH
#define UTIL_CUDA_CUH

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

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
#define CUDA_CHECK(errcode) ::util::cuda_check(errcode, __FILE__, __LINE__)

namespace util
{

/**
 * If code is not cudaSuccess, print an error message and terminate the program.
 */
inline void cuda_check(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << file << ':' << line << ": " << cudaGetErrorString(code) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}   // namespace util

#endif
