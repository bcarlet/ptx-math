#include "model.h"
#include "tuning.h"

#include "tables/ex2_table.h"
#include "common/fpmanip.h"
#include "common/nan.h"
#include "common/squarer.h"

#include <stdbool.h>
#include <math.h>

static int min(int a, int b);
static float fexp2i(float n);

static const m_params model_params =
{
    .table = ex2_table,
    .bias = UINT64_C(0x6fc4000000000000),
    .truncation = 19
};

float model_ex2(float x)
{
    return parameterized_ex2(x, &model_params);
}

float parameterized_ex2(float x, const m_params *params)
{
    switch (fpclassify(x))
    {
    case FP_NAN:
        return canonical_nan();
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

    uint32_t x_bits = reinterpret_float(x);

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
    uint64_t c2_term = c[2] * square_approx(xl, params->truncation);

    c0_term <<= EX2_C0_TERM_ALIGNMENT;
    c1_term <<= EX2_C1_TERM_ALIGNMENT;
    c2_term <<= EX2_C2_TERM_ALIGNMENT;

    uint64_t sum = c0_term + c1_term + c2_term;

    sum += params->bias >> ((64 - EX2_SUM_WEIGHT) + 23);

    const uint32_t r_frac = (sum >> (EX2_SUM_WEIGHT - 23)) & MASK_U32(23);
    const uint32_t r_bits = FP_FORMAT(0u, 127u, r_frac);

    const float r = reinterpret_uint(r_bits);
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
