#ifndef TUNING_COMMON_HPP
#define TUNING_COMMON_HPP

#include <functional>

#include "counters.hpp"

template<class... Args>
using eval_t = std::function<counters (Args...)>;

#endif
