#ifndef BATCHING_HPP
#define BATCHING_HPP

#include <cstddef>

#include "interval.hpp"

std::size_t initialize_batch(const interval &global, float start, float &next,
                             std::size_t max_size, float *x, float *y);

#endif
