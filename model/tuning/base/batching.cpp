#include "batching.hpp"

#include <cmath>

std::size_t initialize_batch(const interval &global, float start, float &next,
                             std::size_t max_size, float *x, float *y)
{
    std::size_t i = 0;
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
