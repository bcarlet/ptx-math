#ifndef TEST_HANDLER_HPP
#define TEST_HANDLER_HPP

#include "base/interval.hpp"
#include "base/fntypes.hpp"
#include "basic_counters.hpp"

using test_fn = std::function<basic_counters(const map_fn &)>;

test_fn make_test_handler(const interval &sub, const map_fn &reference, const sync_fn &sync,
                          float *reference_buf, float *model_buf, std::size_t buf_size);

#endif
