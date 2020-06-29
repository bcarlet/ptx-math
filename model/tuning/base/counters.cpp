#include "counters.hpp"

#include <cstring>
#include <cmath>

void counters::accumulate(std::size_t n, const float *lhs, const float *rhs)
{
    for (std::size_t i = 0; i < n; i++)
    {
        if (memcmp(lhs + i, rhs + i, sizeof(float)) == 0)
        {
            exact++;
        }
        else
        {
            const float absl = fabsf(lhs[i]);
            const float absr = fabsf(rhs[i]);

            if (absl < absr)
                smaller++;
            else if (absl > absr)
                larger++;
        }
    }

    total += n;
}
