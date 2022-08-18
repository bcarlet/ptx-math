#include "bias.hpp"
#include "binsearch.hpp"

#include <limits>

namespace tuning
{

std::pair<bias_t, error> bias_search(const eval_t<bias_t> &eval)
{
    using bs = binsearch<bias_t>;
    using limits = std::numeric_limits<bias_t>;

    bs search(limits::lowest(), limits::max());
    bs::state state;
    error err;

    do
    {
        err = eval(search.point());

        if (err.regions != 1u)
            break;

        state = search.step((err.rightmost == error::NON_NEGATIVE) ? 1 : -1);

    } while (state == bs::CONTINUE);

    return std::make_pair(search.point(), err);
}

}   // namespace tuning
