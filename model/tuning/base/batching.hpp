#ifndef BATCHING_HPP
#define BATCHING_HPP

#include <tuple>

/**
 * Wraps the resulting batch size and the value which will start the next batch.
 */
using batch_result = std::pair<std::size_t, float>;

batch_result initialize_batch(float start, float last, float *x, float *y, std::size_t buf_size);

#endif
