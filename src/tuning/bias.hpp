#ifndef TUNING_BIAS_HPP
#define TUNING_BIAS_HPP

#include <tuple>

#include "model.hpp"
#include "sign_counter.hpp"
#include "testing.hpp"

/**
 * Wraps the resulting bias and the test counters obtained with that bias.
 */
using bias_results = std::pair<uint64_t, sign_counter>;

bias_results bias_search(float first, float last, const tester &t,
                         const model_t<uint64_t> &model);

#endif
