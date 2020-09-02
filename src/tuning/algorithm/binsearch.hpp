#ifndef TUNING_ALGORITHM_BINSEARCH_HPP
#define TUNING_ALGORITHM_BINSEARCH_HPP

#include <cstdint>

enum class bs_state
{
    CONTINUE,
    SUCCESS,
    FAIL
};

bs_state bin_search(uint64_t l, uint64_t r, uint64_t &test);

bs_state bin_search(uint64_t &l, uint64_t &r, uint64_t &test, int test_cmp);

#endif
