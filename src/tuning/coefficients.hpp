#ifndef TUNING_COEFFICIENTS_HPP
#define TUNING_COEFFICIENTS_HPP

#include <array>
#include <tuple>

#include "common.hpp"

namespace tuning
{

using poly_t = std::array<coefficient_t, 3>;

std::tuple<bias_t, poly_t, error>
coefficient_search(const eval_t<bias_t, const poly_t &> &eval,
                   const std::array<int, 3> &signs,
                   const poly_t &initial);

}   // namespace tuning

#endif
