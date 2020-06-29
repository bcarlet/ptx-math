#include "basic_counters.hpp"

#include <cmath>

void basic_counters::accumulate(std::size_t n, const float *gpu, const float *model)
{
    for (std::size_t i = 0; i < n; i++)
    {
        if (gpu[i] == model[i])
        {
            exact++;
            continue;
        }

        const float abs_gpu = fabs(gpu[i]);
        const float abs_model = fabs(model[i]);

        if (abs_model < abs_gpu)
            smaller++;
        else
            larger++;
    }

    total += n;
}

void basic_counters::clear()
{
    smaller = 0;
    larger = 0;
    exact = 0;
    total = 0;
}
