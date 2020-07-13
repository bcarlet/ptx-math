#include "basic_counters.hpp"

#include <cmath>

void basic_counters::accumulate(std::size_t n, const float *ref, const float *model)
{
    for (std::size_t i = 0; i < n; i++)
    {
        if (ref[i] == model[i])
        {
            exact++;
            continue;
        }

        const err_sign sign = (fabs(model[i]) < fabs(ref[i])) ? NEGATIVE : POSITIVE;

        if (last_sign != sign)
        {
            regions++;
            last_sign = sign;
        }
    }

    total += n;
}

void clear(basic_counters &count)
{
    count.exact = 0;
    count.total = 0;
    count.regions = 0;
    count.last_sign = basic_counters::UNDEFINED;
}
