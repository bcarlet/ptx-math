#ifndef COEFFICIENTS_HPP
#define COEFFICIENTS_HPP

#include <array>
#include <tuple>

#include "base/interval.hpp"
#include "base/fntypes.hpp"
#include "basic_counters.hpp"

enum class coeff_sign
{
    NEGATIVE,
    POSITIVE
};

using coeff_arr = std::array<uint32_t, 3>;

/**
 * Wraps the resulting bias and coefficients, and the test counters obtained
 * with those parameters.
 */
using coeff_results = std::tuple<uint64_t, coeff_arr, basic_counters>;

coeff_results coeff_search(const interval &sub, float *gpu_buf, float *model_buf, uint32_t buf_size,
                           const mapf_t &gpu, const genf_t<uint64_t, const coeff_arr &> &model_gen,
                           const syncf_t &sync, const std::array<coeff_sign, 3> &config,
                           const coeff_arr &initial);

#endif
