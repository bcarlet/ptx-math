#ifndef BIAS_HPP
#define BIAS_HPP

#include <tuple>

#include "base/fntypes.hpp"
#include "basic_counters.hpp"
#include "test_handler.hpp"

/**
 * Wraps the resulting bias and the test counters obtained with that bias.
 */
using bias_results = std::pair<uint64_t, basic_counters>;

bias_results bias_search(const testf_t &test, const genf_t<uint64_t> &model_gen);

#endif
