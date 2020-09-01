#include "devices.hpp"
#include "cuda.hpp"

std::vector<cudaDeviceProp> get_device_props()
{
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::vector<cudaDeviceProp> props(device_count);

    for (int device = 0; device < device_count; device++)
    {
        CUDA_CHECK(cudaGetDeviceProperties(props.data() + device, device));
    }

    return props;
}

std::ostream &operator<<(std::ostream &stream, const cudaDeviceProp &prop)
{
    stream << prop.name << " (sm_" << prop.major << prop.minor << ')';

    return stream;
}
