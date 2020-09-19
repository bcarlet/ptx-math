#include "models.h"
#include "tuning.h"

#include "tables/sqrt_table.h"
#include "common/bitcast.h"
#include "common/nan.h"
#include "common/squarer.h"
#include "common/util.h"

#include <math.h>

static const ptxm_params model_params =
{
    .table = ptxm_sqrt_table,
    .bias = UINT64_C(0x7fff800000000000)
};

float ptxm_sqrt_sm5x(float x)
{
    return ptxm_rcp_sm5x(ptxm_rsqrt_sm5x(x));
}

float ptxm_sqrt_sm6x(float x)
{
    return ptxm_sqrt_sm6x_internal(x, &model_params);
}

float ptxm_sqrt_sm6x_internal(float x, const ptxm_params *params)
{
    int x_log2 = 0;

    switch (fpclassify(x))
    {
    case FP_NAN:
        return ptxm_nan();
    case FP_INFINITE:
        return signbit(x) ? ptxm_nan() : INFINITY;
    case FP_ZERO:
        return copysignf(0.0f, x);
    case FP_SUBNORMAL:
        x_log2 = -24;
        x *= 0x1p24f;
        break;
    }

    if (signbit(x)) return ptxm_nan();

    x_log2 += ilogbf(x);

    const uint32_t x_bits = float_as_u32(x);

    const uint32_t xh = UPPER_SIGNIFICAND(x_bits, SQRT_M);
    const uint32_t xl = LOWER_SIGNIFICAND(x_bits, SQRT_M);

    uint32_t index = xh;

    if (x_log2 % 2 != 0)
    {
        x_log2 -= 1;
        index += 1u << SQRT_M;
    }

    const uint32_t *const c = params->table[index];

    uint64_t c0_term = c[0];
    uint64_t c1_term = c[1] * (uint64_t)xl;
    uint64_t c2_term = c[2] * ptxm_square_approx(xl);

    c0_term <<= SQRT_C0_TERM_ALIGNMENT;
    c1_term <<= SQRT_C1_TERM_ALIGNMENT;
    c2_term <<= SQRT_C2_TERM_ALIGNMENT;

    uint64_t sum = c0_term + c1_term - c2_term;

    sum += params->bias >> ((64 - SQRT_SUM_WEIGHT) + 23);

    const uint32_t r_frac = EXTRACT_BITS(sum, 23, SQRT_SUM_WEIGHT);
    const uint32_t r_bits = FP_FORMAT(0u, 127u, r_frac);

    const float r = u32_as_float(r_bits);

    return ldexpf(r, x_log2 / 2);
}
