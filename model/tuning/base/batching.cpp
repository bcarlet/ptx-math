#include "batching.hpp"

#include <cmath>

batch_result initialize_batch(float start, float last, float *x, float *y, std::size_t buf_size)
{
    std::size_t i = 0;
    float val = start;

    while (i < buf_size && val <= last)
    {
        x[i] = val;
        y[i] = val;

        val = nextafterf(val, INFINITY);
        i++;
    }

    return std::make_pair(i, val);
}
