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

static void clear_dirs_lt(direction *dirs, unsigned i)
{
    for (unsigned j = 0; j < i; j++)
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

coeff_results coeff_search(const interval &sub, float *gpu_buf, float *model_buf, uint32_t buf_size,
                           const mapf_t &gpu, const genf_t<uint64_t, const uint32_t (&)[3]> &model_gen,
                           const syncf_t &sync, const coeff_sign (&config)[3], const uint32_t (&initial)[3])
{
    coeff_results results;

    uint64_t &bias = std::get<0>(results);
    uint32_t (&coeff)[3] = std::get<1>(results);
    basic_counters &count = std::get<2>(results);

    for (int i = 0; i < 3; i++)
    {
        coeff[i] = initial[i];
    }

    const auto bs_model_gen = [&coeff, &model_gen](uint64_t bias)
    {
        return model_gen(bias, coeff);
    };

    direction directions[3] = {UNKNOWN, UNKNOWN, UNKNOWN};

    while (true)
    {
        std::tie(bias, count) = bias_search(sub, gpu_buf, model_buf, buf_size, gpu, bs_model_gen, sync);

        if (count.regions == 0 || count.regions > 3)
            break;

        unsigned edit = static_cast<unsigned>(count.regions) - 1u;
        direction dir = get_dir(config[edit], count.last_sign);

        if (directions[edit] == UNKNOWN)
        {
            directions[edit] = dir;
        }
        else if (directions[edit] != dir)
        {
            if (++edit > 2)
                break;

            dir = (directions[edit] != UNKNOWN) ? directions[edit] : DOWN;  // DOWN is arbitrary
        }

        update_coeff(coeff[edit], dir);
        clear_dirs_lt(directions, edit);
    }

    return results;
}
