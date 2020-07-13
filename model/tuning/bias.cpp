#include "bias.hpp"
#include "algorithm/binsearch.hpp"
#include "base/testing.hpp"

bias_results bias_search(const interval &sub, float *ref_buf, float *model_buf, uint32_t buf_size,
                         const mapf_t &ref, const genf_t<uint64_t> &model_gen, const syncf_t &sync)
{
    uint64_t bias;
    basic_counters count;

    uint64_t lower = 0;
    uint64_t upper = UINT64_MAX;

    bs_state state = bin_search(lower, upper, bias);

    while (state == bs_state::CONTINUE)
    {
        clear(count);
        test(sub, ref_buf, model_buf, buf_size, ref, model_gen(bias), sync, count);

        if (count.regions > 1)
            break;

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
            test_cmp = 0;
            break;
        }

        state = bin_search(lower, upper, bias, test_cmp);
    }

    return std::make_pair(bias, count);
}
