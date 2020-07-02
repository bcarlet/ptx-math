#include "bias.hpp"
#include "algorithm/binsearch.hpp"
#include "base/testing.hpp"

bias_results bias_search(const interval &sub, float *gpu_buf, float *model_buf, uint32_t buf_size,
                         const mapf_t &gpu, const genf_t<uint64_t> &model_gen, const syncf_t &sync)
{
    bias_results results;

    uint64_t &bias = results.first;
    basic_counters &count = results.second;

    uint64_t lower = 0;
    uint64_t upper = UINT64_MAX;

    bs_state state = bin_search(lower, upper, bias);

    while (state == bs_state::CONTINUE)
    {
        clear(count);
        test(sub, gpu_buf, model_buf, buf_size, gpu, model_gen(bias), sync, count);

        if (count.regions > 1)
            return results;

        int test_cmp;

        switch (count.last_sign)
        {
        case basic_counters::NEGATIVE:
            test_cmp = -1;
            break;
        case basic_counters::POSITIVE:
            test_cmp = 1;
            break;
        default:
            return results;
        }

        state = bin_search(lower, upper, bias, test_cmp);
    }

    return results;
}
