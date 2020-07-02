#ifndef COEFFICIENTS_HPP
#define COEFFICIENTS_HPP

#include <tuple>

#include "base/interval.hpp"
#include "base/fntypes.hpp"
#include "basic_counters.hpp"

enum class coeff_sign
{
    NEGATIVE,
    POSITIVE
};

/**
 * Wraps the resulting bias and coefficients, and the test counters obtained
 * with those parameters.
 */
using coeff_results = std::tuple<uint64_t, uint32_t [3], basic_counters>;

coeff_results coeff_search(const interval &sub, float *gpu_buf, float *model_buf, uint32_t buf_size,
                           const mapf_t &gpu, const genf_t<uint64_t, const uint32_t (&)[3]> &model_gen,
                           const syncf_t &sync, const coeff_sign (&config)[3], const uint32_t (&initial)[3]);

#endif
