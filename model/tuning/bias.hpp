#ifndef BIAS_HPP
#define BIAS_HPP

#include "base/counters.hpp"
#include "base/interval.hpp"
#include "base/fntypes.hpp"

counters bias_search(const interval &test_space, float *gpubuf, float *modelbuf,
                     uint32_t bufsize, const mapf_t &gpu, const syncf_t &gpusync,
                     const genf_t<uint64_t> &model_gen, uint64_t &bias);

#endif
