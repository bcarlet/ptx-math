#include "devices.hpp"
#include "cuda.hpp"

#include <cstdio>

static int get_device_count()
{
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));

    return count;
}

void print_devices()
{
    const int device_count = get_device_count();

    cudaDeviceProp props;

    for (int device = 0; device < device_count; device++)
    {
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));

        fprintf(stdout, "Device %d: %s (sm_%d%d)\n",
                device, props.name, props.major, props.minor);
    }
}

std::vector<cudaDeviceProp> get_device_props()
{
    const int device_count = get_device_count();

    std::vector<cudaDeviceProp> props(device_count);

    for (int device = 0; device < device_count; device++)
    {
        CUDA_CHECK(cudaGetDeviceProperties(props.data() + device, device));
    }

    return props;
}
