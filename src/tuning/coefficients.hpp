#ifndef TUNING_COEFFICIENTS_HPP
#define TUNING_COEFFICIENTS_HPP

#include <array>
#include <tuple>

#include "base/fntypes.hpp"
#include "basic_counters.hpp"
#include "test_handler.hpp"

enum class coeff_sign
{
    NEGATIVE,
    POSITIVE
};

template<class T> using vec3 = std::array<T, 3>;

/**
 * Wraps the resulting bias and coefficients, and the test counters obtained
 * with those parameters.
 */
using coeff_results = std::tuple<uint64_t, vec3<uint32_t>, basic_counters>;

coeff_results coeff_search(const test_fn &test, const gen_fn<uint64_t, const vec3<uint32_t> &> &model_gen,
                           const vec3<coeff_sign> &config, const vec3<uint32_t> &initial);

#endif
