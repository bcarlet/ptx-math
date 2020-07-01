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

bias_results bias_search(const interval &test_space, float *gpubuf, float *modelbuf,
                         uint32_t bufsize, const mapf_t &gpu, const syncf_t &gpusync,
                         const genf_t<uint64_t> &model_gen);

#endif
