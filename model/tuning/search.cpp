#include "search.hpp"
#include "binsearch.hpp"
#include "testing.hpp"

static int cmp(uint64_t x, uint64_t y)
{
    return (x < y) ? -1 : (x > y);
}

counters bias_search(const interval &test_space, float *gpubuf, float *modelbuf,
                     uint32_t bufsize, const mapf_t &gpu, const syncf_t &gpusync,
                     const genf_t<uint64_t> &model_gen, uint64_t &bias)
{
    counters cnt = {};

    uint64_t lower = 0;
    uint64_t upper = UINT64_MAX;

    bs_state state = bin_search(lower, upper, bias);

    while (state == bs_state::CONTINUE)
    {
        cnt = test(test_space, gpubuf, modelbuf, bufsize, gpu, gpusync, model_gen(bias));

        state = bin_search(lower, upper, bias, cmp(cnt.larger, cnt.smaller));
    }

    return cnt;
}
