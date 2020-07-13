#ifndef TESTING_HPP
#define TESTING_HPP

#include "counters.hpp"
#include "interval.hpp"
#include "fntypes.hpp"

void test(const interval &sub, float *reference_buf, float *model_buf, std::size_t buf_size,
          const map_fn &reference, const map_fn &model, const sync_fn &sync, counters &results);

#endif
