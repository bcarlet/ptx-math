#ifndef TESTING_HPP
#define TESTING_HPP

#include "counters.hpp"
#include "interval.hpp"
#include "fntypes.hpp"

counters test(const interval &test_space, float *gpubuf, float *modelbuf, uint32_t bufsize,
              const mapf_t &gpu, const syncf_t &gpusync, const mapf_t &model);

#endif
