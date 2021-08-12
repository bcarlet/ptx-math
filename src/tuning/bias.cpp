#include "bias.hpp"
#include "binsearch.hpp"

#include <limits>

std::pair<uint64_t, counters> bias_search(const eval_t<uint64_t> &eval)
{
    using bs = binsearch<uint64_t>;
    using limits = std::numeric_limits<uint64_t>;

    counters count;

    bs search(limits::lowest(), limits::max());
    bs::state state = bs::CONTINUE;

    while (state == bs::CONTINUE)
    {
        count = eval(search.point());

        if (count.regions() > 1u)
            break;

        int cmp;

        switch (count.last())
        {
        case counters::NEGATIVE:
            cmp = -1;
            break;
        case counters::POSITIVE:
            cmp = 1;
            break;
        default:
            cmp = 0;
            break;
        }

        state = search.step(cmp);
    }

    return std::make_pair(search.point(), count);
}
