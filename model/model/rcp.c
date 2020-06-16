#include "model.h"
#include "tables/rcp_table.h"
#include "common/fpmanip.h"
#include "common/nan.h"

#include <math.h>

float model_rcp(float x)
{
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
        return 1.0f / x;    // can't handle subnormals explicitly yet
#endif
    }

    const uint32_t x_bits = reinterpret_float(x);

    const uint32_t xh = UPPER_SIGNIFICAND(x_bits, RCP_M);
    const uint32_t xl = LOWER_SIGNIFICAND(x_bits, RCP_M);

    const uint32_t *const c = rcp_table[xh];

    const uint32_t c1_term = c[1] * xl;
    const uint64_t c2_term = (uint64_t)c[2] * (xl * xl);

    const uint64_t c0_aligned = (uint64_t)c[0] << 30;
    const uint64_t c1_term_aligned = (uint64_t)c1_term << 17;

    uint64_t sum = c0_aligned - c1_term_aligned + c2_term;

    sum += UINT64_C(1) << (56 - 24);

    uint32_t r_exp = 127u;

    if (!(sum >> 56))
    {
        r_exp -= 1;
        sum <<= 1;
    }

    const uint32_t r_sig = (sum >> (56 - 23)) & MASK_U32(23);
    const uint32_t r_sign = signbit(x) ? 1u : 0u;

    const uint32_t r_bits = FP_FORMAT(r_sign, r_exp, r_sig);

    const float r = reinterpret_uint(r_bits);

    return ldexpf(r, -ilogbf(x));
}
