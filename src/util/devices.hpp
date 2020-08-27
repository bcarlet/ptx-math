#ifndef UTIL_DEVICES_HPP
#define UTIL_DEVICES_HPP

#include <cuda_runtime.h>
#include <vector>

/**
 * Print the available compute-capable devices to stdout.
 */
void print_devices();

/**
 * Get device properties for all available compute-capable devices.
 */
std::vector<cudaDeviceProp> get_device_props();

#endif
