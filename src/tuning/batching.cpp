#include "batching.hpp"

#include <cmath>

batcher::batcher(float first, float last, float *x, float *y,
                 std::size_t buf_size) :
    next(first),
    last(last),
    x(x),
    y(y),
    buf_size(buf_size)
{
}

std::size_t batcher::init_next()
{
    std::size_t i = 0;

    while (i < buf_size && next <= last)
    {
        x[i] = next;
        y[i] = next;

        next = std::nextafter(next, INFINITY);
        i++;
    }

    return i;
}
