#ifndef TUNING_BIAS_HPP
#define TUNING_BIAS_HPP

#include <cstdint>
#include <utility>

#include "common.hpp"

std::pair<uint64_t, counters> bias_search(const eval_t<uint64_t> &eval);

#endif
