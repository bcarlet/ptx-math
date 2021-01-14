#include "models.h"
#include "tuning.h"

#include "tables/sin_table.h"
#include "reduction/rro_sincos.h"
#include "common/bitcast.h"
#include "common/nan.h"
#include "common/squarer.h"
#include "common/util.h"

#define RRO_SM5X ptxm_rro_sincos_sm5x
#define RRO_SM70 ptxm_rro_sincos_sm70

static float mufu_sin(uint32_t, const ptxm_params *);
static float mufu_cos(uint32_t);
static float poly_sincos(uint32_t, uint32_t, const ptxm_params *);

static const ptxm_params model_params =
{
    .table = ptxm_sin_table,
    .bias = UINT64_C(0x0000000000000000)
};

float ptxm_sin_sm5x(float x) { return mufu_sin(RRO_SM5X(x), &model_params); }
float ptxm_sin_sm70(float x) { return mufu_sin(RRO_SM70(x), &model_params); }

float ptxm_cos_sm5x(float x) { return mufu_cos(RRO_SM5X(x)); }
float ptxm_cos_sm70(float x) { return mufu_cos(RRO_SM70(x)); }

float ptxm_sin_sm5x_internal(float x, const ptxm_params *params)
{
    return mufu_sin(RRO_SM5X(x), params);
}

float mufu_sin(uint32_t reduced, const ptxm_params *params)
{
    uint32_t sign = reduced >> 31;
    uint32_t quadrant = (reduced >> 23) & MASK_U32(8);

    if (quadrant & 0x80u) return ptxm_nan();

    if (quadrant & 1u) reduced = ~reduced;
    if (quadrant & 2u) sign = !sign;

    return poly_sincos(reduced, sign, params);
}

float mufu_cos(uint32_t reduced)
{
    uint32_t sign = reduced >> 31;
    uint32_t quadrant = (reduced >> 23) & MASK_U32(8);

    if (quadrant & 0x80u) return ptxm_nan();

    sign = 0u;

    if (!(quadrant & 1u)) reduced = ~reduced;
    if ((quadrant + 1u) & 2u) sign = 1u;

    return poly_sincos(reduced, sign, &model_params);
}

float poly_sincos(uint32_t reduced, uint32_t sign, const ptxm_params *params)
{
    const uint32_t xh = UPPER_SIGNIFICAND(reduced, SIN_M);
    const uint32_t xl = LOWER_SIGNIFICAND(reduced, SIN_M);

    const uint32_t *const c = params->table[xh];

    uint64_t c0_term = c[0];
    uint64_t c1_term = c[1] * (uint64_t)xl;
    uint64_t c2_term = c[2] * ptxm_square_approx(xl);

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

    const uint32_t r_frac = EXTRACT_BITS(sum, 23, SIN_SUM_WEIGHT);
    const uint32_t r_bits = FP_FORMAT(sign, r_exp, r_frac);

    return u32_as_float(r_bits);
}
