#include "cuda.hpp"

#include <cstdlib>
#include <iostream>

void cuda_error_check(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA error in file " << file << " on line " << line
                  << ": " << cudaGetErrorString(code) << '\n';

        std::exit(-1);
    }
}
