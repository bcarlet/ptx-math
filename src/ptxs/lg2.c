#include "models.h"
#include "tuning.h"

#include "tables/lg2_table.h"
#include "common/bitcast.h"
#include "common/fpmanip.h"
#include "common/nan.h"
#include "common/squarer.h"

#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define LG2_SUM_TRUNCATION 20

static const ptxs_params model_params =
{
    .table = ptxs_lg2_table,
    .bias = UINT64_C(0x868b000000000000)
};

float ptxs_lg2(float x)
{
    return ptxs_param_lg2(x, &model_params);
}

float ptxs_param_lg2(float x, const ptxs_params *params)
{
    bool subnormal = false;

    switch (fpclassify(x))
    {
    case FP_NAN:
        return ptxs_nan();
    case FP_INFINITE:
        return signbit(x) ? ptxs_nan() : INFINITY;
    case FP_ZERO:
        return -INFINITY;
    case FP_SUBNORMAL:
        x *= 0x1p24f;
        subnormal = true;
        break;
    }

    if (signbit(x)) return ptxs_nan();
    if (x == 1.0f) return 0.0f;

    const uint32_t x_bits = float_as_u32(x);

    const uint32_t xh = UPPER_SIGNIFICAND(x_bits, LG2_M);
    const uint32_t xl = LOWER_SIGNIFICAND(x_bits, LG2_M);

    const uint32_t *const c = params->table[xh];

    uint64_t c0_term = c[0];
    uint64_t c1_term = c[1] * (uint64_t)xl;
    uint64_t c2_term = c[2] * ptxs_square_approx(xl);

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

    const float r = u32_as_float(r_bits);

    return subnormal ? (r - 24.0f) : r;
}
