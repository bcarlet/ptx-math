#ifndef BATCHING_HPP
#define BATCHING_HPP

#include <cstdint>

struct interval
{
    float least, greatest;
};

uint32_t initialize_batch(const interval &global, float start, float &next,
                          uint32_t max_size, float *x, float *y);

#endif
