#include "models.h"
#include "tuning.h"

#include "tables/ex2_table.h"
#include "common/bitcast.h"
#include "common/nan.h"
#include "common/squarer.h"
#include "common/util.h"

#include <stdbool.h>
#include <math.h>

static int min(int a, int b);
static float fexp2i(float n);

static const ptxs_params model_params =
{
    .table = ptxs_ex2_table,
    .bias = UINT64_C(0x6fc4000000000000)
};

float ptxs_ex2(float x)
{
    return ptxs_param_ex2(x, &model_params);
}

float ptxs_param_ex2(float x, const ptxs_params *params)
{
    switch (fpclassify(x))
    {
    case FP_NAN:
        return ptxs_nan();
    case FP_INFINITE:
        return signbit(x) ? 0.0f : INFINITY;
    case FP_ZERO:
    case FP_SUBNORMAL:
        return 1.0f;
    }

    bool square_result = false;

    if (x < -126.0f)
    {
        x *= 0.5f;
        square_result = true;
    }

    float integral;
    x = modff(x, &integral);

    uint32_t x_bits = float_as_u32(x);

    if (x != 0.0f)
    {
        x_bits |= UINT32_C(1) << 23;
        x_bits &= MASK_U32(24);

        x_bits >>= min(-ilogbf(x), 24);

        if (x < 0.0f && x_bits != 0u)
        {
            x_bits = ~x_bits;
            integral -= 1.0f;
        }
    }

    const uint32_t xh = UPPER_SIGNIFICAND(x_bits, EX2_M);
    const uint32_t xl = LOWER_SIGNIFICAND(x_bits, EX2_M);

    const uint32_t *const c = params->table[xh];

    uint64_t c0_term = c[0];
    uint64_t c1_term = c[1] * (uint64_t)xl;
    uint64_t c2_term = c[2] * ptxs_square_approx(xl);

    c0_term <<= EX2_C0_TERM_ALIGNMENT;
    c1_term <<= EX2_C1_TERM_ALIGNMENT;
    c2_term <<= EX2_C2_TERM_ALIGNMENT;

    uint64_t sum = c0_term + c1_term + c2_term;

    sum += params->bias >> ((64 - EX2_SUM_WEIGHT) + 23);

    const uint32_t r_frac = EXTRACT_BITS(sum, 23, EX2_SUM_WEIGHT);
    const uint32_t r_bits = FP_FORMAT(0u, 127u, r_frac);

    const float r = u32_as_float(r_bits);
    const float reconstr = fexp2i(integral) * r;

    return square_result ? (reconstr * reconstr) : reconstr;
}

int min(int a, int b)
{
    return (a < b) ? a : b;
}

float fexp2i(float n)
{
    if (n > 127.0f) return INFINITY;
    if (n < -126.0f) return 0.0f;

    return ldexpf(1.0f, (int)n);
}
