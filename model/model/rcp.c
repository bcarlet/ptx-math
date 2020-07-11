#include "model.h"
#include "tuning.h"

#include "tables/rcp_table.h"
#include "common/fpmanip.h"
#include "common/nan.h"
#include "common/squarer.h"

#include <math.h>

static const m_params model_params =
{
    .table = rcp_table,
    .bias = UINT64_C(0x67e7000000000000),
    .truncation = 17
};

float model_rcp(float x)
{
    return parameterized_rcp(x, &model_params);
}

float parameterized_rcp(float x, const m_params *params)
{
    int x_log2;

    switch (fpclassify(x))
    {
    case FP_NAN:
        return canonical_nan();
    case FP_INFINITE:
        return copysignf(0.0f, x);
    case FP_ZERO:
        return copysignf(INFINITY, x);
#ifdef PTX_FTZ
    case FP_SUBNORMAL:
        return copysignf(INFINITY, x);
#else
    case FP_SUBNORMAL:
        x_log2 = ilogbf(x);
        x *= 0x1p24f;
        break;
#endif
    case FP_NORMAL:
        x_log2 = ilogbf(x);
        break;
    }

    const uint32_t x_bits = reinterpret_float(x);

    const uint32_t xh = UPPER_SIGNIFICAND(x_bits, RCP_M);
    const uint32_t xl = LOWER_SIGNIFICAND(x_bits, RCP_M);

    const uint32_t *const c = params->table[xh];

    uint64_t c0_term = c[0];
    uint64_t c1_term = c[1] * xl;   // won't exceed 32 bits
    uint64_t c2_term = c[2] * square_approx(xl, params->truncation);

    c0_term <<= RCP_C0_TERM_ALIGNMENT;
    c1_term <<= RCP_C1_TERM_ALIGNMENT;
    c2_term <<= RCP_C2_TERM_ALIGNMENT;

    uint64_t sum = c0_term - c1_term + c2_term;

    sum += params->bias >> ((64 - RCP_SUM_WEIGHT) + 23);

    uint32_t r_exp = 127u;

    if (!(sum >> RCP_SUM_WEIGHT))
    {
        r_exp -= 1u;
        sum <<= 1;
    }

    const uint32_t r_frac = (sum >> (RCP_SUM_WEIGHT - 23)) & MASK_U32(23);
    const uint32_t r_sign = signbit(x) ? 1u : 0u;

    const uint32_t r_bits = FP_FORMAT(r_sign, r_exp, r_frac);

    const float r = reinterpret_uint(r_bits);

    return ldexpf(r, -x_log2);
}
