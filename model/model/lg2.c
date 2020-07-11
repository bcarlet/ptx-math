#include "model.h"
#include "tuning.h"

#include "tables/lg2_table.h"
#include "common/fpmanip.h"
#include "common/nan.h"
#include "common/squarer.h"

#include <stdbool.h>
#include <math.h>

#define LG2_SUM_TRUNCATION 20

static const m_params default_params =
{
    .table = lg2_table,
    .bias = UINT64_C(0x868b000000000000),
    .truncation = 19
};

float model_lg2(float x)
{
    return parameterized_lg2(x, &default_params);
}

float parameterized_lg2(float x, const m_params *params)
{
    bool subnormal = false;

    switch (fpclassify(x))
    {
    case FP_NAN:
        return canonical_nan();
    case FP_INFINITE:
        return signbit(x) ? canonical_nan() : INFINITY;
    case FP_ZERO:
        return -INFINITY;
#ifdef PTX_FTZ
    case FP_SUBNORMAL:
        return -INFINITY;
#else
    case FP_SUBNORMAL:
        x *= 0x1p24f;
        subnormal = true;

        break;
#endif
    case FP_NORMAL:
        if (x == 1.0f) return 0.0f;

        break;
    }

    if (signbit(x)) return canonical_nan();

    const uint32_t x_bits = reinterpret_float(x);

    const uint32_t xh = UPPER_SIGNIFICAND(x_bits, LG2_M);
    const uint32_t xl = LOWER_SIGNIFICAND(x_bits, LG2_M);

    const uint32_t *const c = params->table[xh];

    uint64_t c0_term = c[0];
    uint64_t c1_term = c[1] * (uint64_t)xl;
    uint64_t c2_term = c[2] * square_approx(xl, params->truncation);

    c0_term <<= LG2_C0_TERM_ALIGNMENT;
    c1_term <<= LG2_C1_TERM_ALIGNMENT;
    c2_term <<= LG2_C2_TERM_ALIGNMENT;

    uint64_t sum = c0_term + c1_term - c2_term;

    sum += params->bias >> ((64 - LG2_SUM_WEIGHT) + 23);

    const int64_t offset = (int64_t)ilogbf(x) << LG2_SUM_WEIGHT;

    sum = llabs((int64_t)sum + offset);
    sum &= ~MASK_U64(LG2_SUM_TRUNCATION);

    uint32_t r_exp = 127u;

    if (!(sum >> LG2_SUM_WEIGHT))
    {
        do
        {
            r_exp -= 1u;
            sum <<= 1;
        } while (!(sum >> LG2_SUM_WEIGHT));
    }
    else
    {
        while (sum >> (LG2_SUM_WEIGHT + 1))
        {
            r_exp += 1u;
            sum >>= 1;
        }
    }

    const uint32_t r_frac = (sum >> (LG2_SUM_WEIGHT - 23)) & MASK_U32(23);
    const uint32_t r_sign = (x < 1.0f) ? 1u : 0u;

    const uint32_t r_bits = FP_FORMAT(r_sign, r_exp, r_frac);

    const float r = reinterpret_uint(r_bits);

    return subnormal ? (r - 24.0f) : r;
}
