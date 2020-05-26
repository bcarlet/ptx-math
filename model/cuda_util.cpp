#include "cuda_util.hpp"

#include <cstdlib>
#include <cstdio>

void print_devices()
{
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    cudaDeviceProp props;

    for (int device = 0; device < device_count; device++)
    {
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));

        printf("Device %d: %s\n", device, props.name);
    }
}

void cuda_err_check(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in file %s on line %d: %s\n",
                file, line, cudaGetErrorString(code));

        exit(-1);
    }
}
