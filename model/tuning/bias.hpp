#ifndef BIAS_HPP
#define BIAS_HPP

#include <tuple>

#include "base/interval.hpp"
#include "base/fntypes.hpp"
#include "basic_counters.hpp"

/**
 * Wraps the resulting bias and the test counters obtained with that bias.
 */
using bias_results = std::pair<uint64_t, basic_counters>;

bias_results bias_search(const interval &sub, float *gpu_buf, float *model_buf, uint32_t buf_size,
                         const mapf_t &gpu, const genf_t<uint64_t> &model_gen, const syncf_t &sync);

#endif
