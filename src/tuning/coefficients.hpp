#ifndef TUNING_COEFFICIENTS_HPP
#define TUNING_COEFFICIENTS_HPP

#include <array>
#include <tuple>

#include "model.hpp"
#include "sign_counter.hpp"
#include "testing.hpp"

enum class coeff_sign
{
    NEGATIVE,
    POSITIVE
};

template<class T>
using vec3 = std::array<T, 3>;

/**
 * Wraps the resulting bias and coefficients, and the test counters obtained
 * with those parameters.
 */
using coeff_results = std::tuple<uint64_t, vec3<uint32_t>, sign_counter>;

coeff_results coeff_search(float first, float last, const tester &t,
                           const model_t<uint64_t, const vec3<uint32_t> &> &model,
                           const vec3<coeff_sign> &config,
                           const vec3<uint32_t> &initial);

#endif
