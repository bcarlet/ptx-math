#include "model.h"
#include "tuning.h"

#include "tables/rsqrt_table.h"
#include "common/fpmanip.h"
#include "common/nan.h"
#include "common/squarer.h"

#include <math.h>

static const m_params default_params =
{
    .table = rsqrt_table,
    .bias = UINT64_C(0x8000000000000000),
    .truncation = 19
};

float model_rsqrt(float x)
{
    return parameterized_rsqrt(x, &default_params);
}

float parameterized_rsqrt(float x, const m_params *params)
{
    int x_log2;

    switch (fpclassify(x))
    {
    case FP_NAN:
        return canonical_nan();
    case FP_INFINITE:
        return signbit(x) ? canonical_nan() : 0.0f;
    case FP_ZERO:
        return copysignf(INFINITY, x);
#ifdef PTX_FTZ
    case FP_SUBNORMAL:
        return copysignf(INFINITY, x);
#else
    case FP_SUBNORMAL:
        x_log2 = ilogbf(x);
        x = ldexpf(x, -x_log2);     // normalize x
        break;
#endif
    case FP_NORMAL:
        x_log2 = ilogbf(x);
        break;
    }

    if (signbit(x)) return canonical_nan();

    const uint32_t x_bits = reinterpret_float(x);

    const uint32_t xh = UPPER_SIGNIFICAND(x_bits, RSQRT_M);
    const uint32_t xl = LOWER_SIGNIFICAND(x_bits, RSQRT_M);

    uint32_t index = xh;

    if (x_log2 % 2 != 0)
    {
        x_log2 -= 1;
        index += 1u << RSQRT_M;
    }

    const uint32_t *const c = params->table[index];

    uint64_t c0_term = c[0];
    uint64_t c1_term = c[1] * (uint64_t)xl;
    uint64_t c2_term = c[2] * square_approx(xl, params->truncation);

    c0_term <<= RSQRT_C0_TERM_ALIGNMENT;
    c1_term <<= RSQRT_C1_TERM_ALIGNMENT;
    c2_term <<= RSQRT_C2_TERM_ALIGNMENT;

    uint64_t sum = c0_term - c1_term + c2_term;

    sum += params->bias >> ((64 - RSQRT_SUM_WEIGHT) + 23);

    uint32_t r_exp = 127u;

    if (!(sum >> RSQRT_SUM_WEIGHT))
    {
        r_exp -= 1u;
        sum <<= 1;
    }

    const uint32_t r_frac = (sum >> (RSQRT_SUM_WEIGHT - 23)) & MASK_U32(23);
    const uint32_t r_sign = signbit(x) ? 1u : 0u;

    const uint32_t r_bits = FP_FORMAT(r_sign, r_exp, r_frac);

    const float r = reinterpret_uint(r_bits);

    return ldexpf(r, -x_log2 / 2);
}
