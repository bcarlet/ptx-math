#include "coefficients.hpp"
#include "bias.hpp"

namespace tuning
{

std::tuple<bias_t, poly_t, error>
coefficient_search(const eval_t<bias_t, const poly_t &> &eval,
                   const std::array<int, 3> &signs,
                   const poly_t &initial)
{
    bias_t bias;
    poly_t coefficients = initial;
    error err;

    const eval_t<bias_t> bias_eval = [&coefficients, &eval](bias_t bias)
    {
        return eval(bias, coefficients);
    };

    std::array<int, 3> directions = {};

    while (true)
    {
        std::tie(bias, err) = bias_search(bias_eval);

        if (err.regions == 0u || err.regions > 3u)
            break;

        count_t edit = err.regions - 1u;
        int dir = (err.rightmost == error::NON_NEGATIVE) ? -1 : 1;

        if (directions[edit] == 0)
        {
            directions[edit] = dir;
        }
        else if (directions[edit] != dir)
        {
            if (++edit > 2u)
                break;
        }

        dir = directions[edit] * signs[edit];
        coefficients[edit] += static_cast<coefficient_t>((dir >= 0) ? 1 : -1);

        for (count_t i = 0; i < edit; i++)
        {
            directions[i] = 0;
        }
    }

    return std::make_tuple(bias, coefficients, err);
}

}   // namespace tuning
