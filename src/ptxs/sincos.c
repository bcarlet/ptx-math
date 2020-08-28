#include "models.h"
#include "tuning.h"

#include "reduction/rro_sincos.h"
#include "tables/sin_table.h"
#include "common/bitcast.h"
#include "common/fpmanip.h"
#include "common/nan.h"
#include "common/squarer.h"

static uint32_t rro_internal(float, uint32_t *, uint32_t *);
static float poly_sincos(uint32_t, uint32_t, const ptxs_params *);

static const ptxs_params model_params =
{
    .table = sin_table,
    .bias = UINT64_C(0x0000000000000000),
    .truncation = 19
};

float ptxs_sin(float x)
{
    return ptxs_param_sin(x, &model_params);
}

float ptxs_param_sin(float x, const ptxs_params *params)
{
    uint32_t sign, quadrant;
    uint32_t reduced = rro_internal(x, &sign, &quadrant);

    if (quadrant >> 7) return canonical_nan();

    if (quadrant & 1u) reduced = ~reduced;
    if (quadrant & 2u) sign = !sign;

    return poly_sincos(reduced, sign, params);
}

float ptxs_cos(float x)
{
    uint32_t sign, quadrant;
    uint32_t reduced = rro_internal(x, &sign, &quadrant);

    if (quadrant >> 7) return canonical_nan();

    sign = 0u;

    if (!(quadrant & 1u)) reduced = ~reduced;
    if ((quadrant + 1u) & 2u) sign = 1u;

    return poly_sincos(reduced, sign, &model_params);
}

uint32_t rro_internal(float x, uint32_t *sign, uint32_t *quadrant)
{
    const uint32_t rr = rro_sincos(x);

    *sign = rr >> 31;
    *quadrant = (rr >> 23) & MASK_U32(8);

    return rr;
}

float poly_sincos(uint32_t reduced, uint32_t sign, const ptxs_params *params)
{
    const uint32_t xh = UPPER_SIGNIFICAND(reduced, SIN_M);
    const uint32_t xl = LOWER_SIGNIFICAND(reduced, SIN_M);

    const uint32_t *const c = params->table[xh];

    uint64_t c0_term = c[0];
    uint64_t c1_term = c[1] * (uint64_t)xl;
    uint64_t c2_term = c[2] * square_approx(xl, params->truncation);

    c0_term <<= SIN_C0_TERM_ALIGNMENT;
    c1_term <<= SIN_C1_TERM_ALIGNMENT;
    c2_term <<= SIN_C2_TERM_ALIGNMENT;

    uint64_t sum = c0_term + c1_term - c2_term;

    sum += params->bias >> ((64 - SIN_SUM_WEIGHT) + 23);

    uint32_t r_exp = 0u;

    if (sum)
    {
        r_exp = 127u;

        while (!(sum >> SIN_SUM_WEIGHT))
        {
            r_exp -= 1u;
            sum <<= 1;
        }
    }

    const uint32_t r_frac = (sum >> (SIN_SUM_WEIGHT - 23)) & MASK_U32(23);
    const uint32_t r_bits = FP_FORMAT(sign, r_exp, r_frac);

    return u32_as_float(r_bits);
}
