#ifndef UTIL_DEVICES_HPP
#define UTIL_DEVICES_HPP

#include <cuda_runtime.h>
#include <ostream>
#include <vector>

/**
 * Get device properties for all available compute-capable devices. Device 0 is
 * stored at index 0 of the returned vector, and so on.
 */
std::vector<cudaDeviceProp> get_device_props();

std::ostream &operator<<(std::ostream &stream, const cudaDeviceProp &prop);

#endif
