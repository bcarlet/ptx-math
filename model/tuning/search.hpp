#ifndef SEARCH_HPP
#define SEARCH_HPP

#include "counters.hpp"
#include "interval.hpp"
#include "fntypes.hpp"

counters bias_search(const interval &test_space, float *gpubuf, float *modelbuf,
                     uint32_t bufsize, const mapf_t &gpu, const syncf_t &gpusync,
                     const genf_t<uint64_t> &model_gen, uint64_t &bias);

#endif
