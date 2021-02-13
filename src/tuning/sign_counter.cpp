#include "sign_counter.hpp"

#include <cmath>

void sign_counter::accumulate(std::size_t n, const float *reference, const float *model)
{
    for (std::size_t i = 0; i < n; i++)
    {
        if (reference[i] == model[i])
        {
            exact++;
            continue;
        }

        const error_sign sign = std::abs(model[i]) < std::abs(reference[i]) ? NEGATIVE : POSITIVE;

        if (last_sign != sign)
        {
            regions++;
            last_sign = sign;
        }
    }

    total += n;
}

void sign_counter::clear()
{
    exact = 0;
    total = 0;
    regions = 0;
    last_sign = UNDEFINED;
}

sign_counter::error_sign sign_counter::first() const
{
    if (last_sign == UNDEFINED)
        return UNDEFINED;

    if (regions % 2 == 0)
        return last_sign == POSITIVE ? NEGATIVE : POSITIVE;
    else
        return last_sign;
}

sign_counter::error_sign sign_counter::last() const
{
    return last_sign;
}
