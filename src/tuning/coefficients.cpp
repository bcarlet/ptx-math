#include "coefficients.hpp"
#include "bias.hpp"

#include <cstddef>

enum direction
{
    DOWN,
    UP,
    UNKNOWN
};

std::tuple<uint64_t, std::array<uint32_t, 3>, counters>
coefficient_search(const eval_t<uint64_t, const std::array<uint32_t, 3> &> &eval,
                   const std::array<bool, 3> &negated,
                   const std::array<uint32_t, 3> &initial)
{
    counters count;

    uint64_t bias;
    std::array<uint32_t, 3> coefficients = initial;

    const auto bias_eval = [&coefficients, &eval](uint64_t bias) -> counters
    {
        return eval(bias, coefficients);
    };

    std::array<direction, 3> directions = {UNKNOWN, UNKNOWN, UNKNOWN};

    while (true)
    {
        std::tie(bias, count) = bias_search(bias_eval);

        if (count.regions() == 0u)
            break;

        if (count.regions() > 3u)
            break;

        std::size_t edit = count.regions() - 1u;
        direction dir = (count.last() == counters::NEGATIVE) ? UP : DOWN;

        if (negated[edit])
            dir = (dir == DOWN) ? UP : DOWN;

        if (directions[edit] == UNKNOWN)
        {
            directions[edit] = dir;
        }
        else if (directions[edit] != dir)
        {
            if (++edit > 2u)
                break;
        }

        if (directions[edit] == DOWN)
            coefficients[edit]--;
        else
            coefficients[edit]++;   // UP is arbitrary for UNKNOWN case

        for (std::size_t i = 0; i < edit; i++)
        {
            directions[i] = UNKNOWN;
        }
    }

    return std::make_tuple(bias, coefficients, count);
}
