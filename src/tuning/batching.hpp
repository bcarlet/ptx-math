#ifndef TUNING_BATCHING_HPP
#define TUNING_BATCHING_HPP

#include <cstddef>

class batcher
{
public:
    batcher(float first, float last, float *x, float *y, std::size_t buf_size);

    std::size_t init_next();

private:
    float next, last;
    float *x, *y;
    std::size_t buf_size;
};

#endif
