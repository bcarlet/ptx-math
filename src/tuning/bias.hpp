#ifndef TUNING_BIAS_HPP
#define TUNING_BIAS_HPP

#include <utility>

#include "common.hpp"

namespace tuning
{

std::pair<bias_t, error> bias_search(const eval_t<bias_t> &eval);

}   // namespace tuning

#endif
