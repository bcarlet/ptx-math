#include "coefficients.hpp"
#include "bias.hpp"

enum direction
{
    DOWN, UP, UNKNOWN
};

static direction get_dir(coeff_sign config, sign_counter::error_sign last)
{
    const bool config_is_neg = (config == coeff_sign::NEGATIVE);
    const bool last_is_neg = (last == sign_counter::NEGATIVE);

    return (config_is_neg != last_is_neg) ? UP : DOWN;
}

coeff_results coeff_search(float first, float last, const tester &t,
                           const model_t<uint64_t, const vec3<uint32_t> &> &model,
                           const vec3<coeff_sign> &config,
                           const vec3<uint32_t> &initial)
{
    uint64_t bias;
    vec3<uint32_t> coeff = initial;
    sign_counter count;

    const auto bs_model = [&coeff, &model](uint64_t bias) -> tester::map_t
    {
        return model(bias, coeff);
    };

    vec3<direction> directions = {UNKNOWN, UNKNOWN, UNKNOWN};

    while (true)
    {
        std::tie(bias, count) = bias_search(first, last, t, bs_model);

        if (count.regions == 0 || count.regions > 3)
            break;

        std::size_t edit = count.regions - 1;

        direction dir = get_dir(config[edit], count.last());

        if (directions[edit] == UNKNOWN)
        {
            directions[edit] = dir;
        }
        else if (directions[edit] != dir)
        {
            if (++edit > 2)
                break;

            dir = directions[edit] == UP ? UP : DOWN;     // DOWN is arbitrary for UNKNOWN case
        }

        if (dir == DOWN)
            coeff[edit]--;
        else
            coeff[edit]++;

        for (std::size_t i = 0; i < edit; i++)
        {
            directions[i] = UNKNOWN;
        }
    }

    return std::make_tuple(bias, coeff, count);
}
