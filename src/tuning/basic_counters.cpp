#include "basic_counters.hpp"

#include <cmath>

void basic_counters::accumulate(std::size_t n, const float *reference, const float *model)
{
    for (std::size_t i = 0; i < n; i++)
    {
        if (reference[i] == model[i])
        {
            exact++;
            continue;
        }

        const err_sign sign = (fabs(model[i]) < fabs(reference[i])) ? NEGATIVE : POSITIVE;

        if (last_sign != sign)
        {
            regions++;
            last_sign = sign;
        }
    }

    total += n;
}
