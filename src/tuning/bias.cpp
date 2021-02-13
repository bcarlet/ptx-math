#include "bias.hpp"
#include "binsearch.hpp"

bias_results bias_search(float first, float last, const tester &t,
                         const model_t<uint64_t> &model)
{
    uint64_t bias;
    sign_counter count;

    using bs = binsearch<uint64_t>;

    bs search(0, UINT64_MAX);
    bs::state state = bs::CONTINUE;

    while (state == bs::CONTINUE)
    {
        bias = search.test_point();

        count.clear();
        t.test(first, last, model(bias), count);

        if (count.regions > 1)
            break;

        int cmp;

        switch (count.last())
        {
        case sign_counter::NEGATIVE:
            cmp = -1;
            break;
        case sign_counter::POSITIVE:
            cmp = 1;
            break;
        default:
            cmp = 0;
            break;
        }

        state = search.step(cmp);
    }

    return std::make_pair(bias, count);
}
