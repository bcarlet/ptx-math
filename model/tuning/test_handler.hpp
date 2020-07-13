#ifndef TEST_HANDLER_HPP
#define TEST_HANDLER_HPP

#include "base/interval.hpp"
#include "base/fntypes.hpp"
#include "basic_counters.hpp"

using testf_t = std::function<basic_counters(const mapf_t &)>;

testf_t make_test_handler(const interval &sub, const mapf_t &ref, const syncf_t &sync,
                          float *ref_buf, float *model_buf, std::size_t buf_size);

#endif
