#ifndef TUNING_MODEL_HPP
#define TUNING_MODEL_HPP

#include "testing.hpp"

template<class... Args>
using model_t = std::function<tester::map_t (Args...)>;

#endif
