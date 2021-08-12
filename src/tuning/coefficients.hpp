#ifndef TUNING_COEFFICIENTS_HPP
#define TUNING_COEFFICIENTS_HPP

#include <array>
#include <cstdint>
#include <tuple>

#include "common.hpp"

std::tuple<uint64_t, std::array<uint32_t, 3>, counters>
coefficient_search(const eval_t<uint64_t, const std::array<uint32_t, 3> &> &eval,
                   const std::array<bool, 3> &negated,
                   const std::array<uint32_t, 3> &initial);

#endif
