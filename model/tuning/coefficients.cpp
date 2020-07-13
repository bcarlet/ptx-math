#include "coefficients.hpp"
#include "bias.hpp"

enum direction
{
    DOWN, UP, UNKNOWN
};

static direction get_dir(coeff_sign config, basic_counters::err_sign last)
{
    const bool config_is_neg = (config == coeff_sign::NEGATIVE);
    const bool last_is_neg = (last == basic_counters::NEGATIVE);

    return (config_is_neg != last_is_neg) ? UP : DOWN;
}

static void clear_dirs_lt(vec3<direction> &dirs, std::size_t i)
{
    for (std::size_t j = 0; j < i; j++)
    {
        dirs[j] = UNKNOWN;
    }
}

static void update_coeff(uint32_t &coeff, direction dir)
{
    if (dir == DOWN)
        coeff--;
    else
        coeff++;
}

coeff_results coeff_search(const interval &sub, float *ref_buf, float *model_buf, uint32_t buf_size,
                           const mapf_t &ref, const genf_t<uint64_t, const vec3<uint32_t> &> &model_gen,
                           const syncf_t &sync, const vec3<coeff_sign> &config, const vec3<uint32_t> &initial)
{
    uint64_t bias;
    vec3<uint32_t> coeff = initial;
    basic_counters count;

    const auto bs_model_gen = [&coeff, &model_gen](uint64_t bias) -> mapf_t
    {
        return model_gen(bias, coeff);
    };

    vec3<direction> directions = {UNKNOWN, UNKNOWN, UNKNOWN};

    while (true)
    {
        std::tie(bias, count) = bias_search(sub, ref_buf, model_buf, buf_size, ref, bs_model_gen, sync);

        if (count.regions == 0 || count.regions > 3)
            break;

        std::size_t edit = count.regions - 1u;
        direction dir = get_dir(config[edit], count.last_sign);

        if (directions[edit] == UNKNOWN)
        {
            directions[edit] = dir;
        }
        else if (directions[edit] != dir)
        {
            if (++edit > 2u)
                break;

            dir = (directions[edit] != UNKNOWN) ? directions[edit] : DOWN;  // DOWN is arbitrary
        }

        update_coeff(coeff[edit], dir);
        clear_dirs_lt(directions, edit);
    }

    return std::make_tuple(bias, coeff, count);
}
