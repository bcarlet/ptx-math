#include "cuda.hpp"

#include <cstdlib>
#include <cstdio>

void cuda_error_check(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in file %s on line %d: %s\n",
                file, line, cudaGetErrorString(code));

        exit(-1);
    }
}