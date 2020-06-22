#include "batching.hpp"

#include <cmath>

uint32_t initialize_batch(const interval &global, float start, float &next,
                          uint32_t max_size, float *x, float *y)
{
    uint32_t i = 0;
    float val = start;

    while (i < max_size && val <= global.greatest)
    {
        x[i] = val;
        y[i] = val;

        val = nextafterf(val, INFINITY);
        i++;
    }

    next = val;

    return i;
}
