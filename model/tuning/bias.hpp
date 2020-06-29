#ifndef BIAS_HPP
#define BIAS_HPP

#include "base/interval.hpp"
#include "base/fntypes.hpp"
#include "basic_counters.hpp"

struct bias_results
{
    uint64_t bias;
    basic_counters count;
};

bias_results bias_search(const interval &test_space, float *gpubuf, float *modelbuf,
                         uint32_t bufsize, const mapf_t &gpu, const syncf_t &gpusync,
                         const genf_t<uint64_t> &model_gen);

#endif
