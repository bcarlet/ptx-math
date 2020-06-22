#ifndef TESTING_HPP
#define TESTING_HPP

#include <functional>

#include "batching.hpp"
#include "counters.hpp"

using mapf_t = std::function<void(int, float *)>;
using syncf_t = std::function<void()>;

counters test(const interval &test_space, float *gpubuf, float *modelbuf, uint32_t bufsize,
              const mapf_t &gpu, const syncf_t &gpusync, const mapf_t &model);

#endif
