#include "models.h"

#include "common/bitcast.h"
#include "common/nan.h"
#include "common/util.h"

#include <math.h>

double ptxm_rsqrt_ftz_f64_sm5x(double x)
{
    x = u64_as_double(double_as_u64(x) & ~MASK_U64(32));

    switch (fpclassify(x))
    {
    case FP_NAN:
        return ptxm_double_nan();
    case FP_INFINITE:
        return signbit(x) ? ptxm_double_nan() : 0.0;
    case FP_ZERO:
    case FP_SUBNORMAL:
        return copysign(INFINITY, x);
    }

    if (signbit(x)) return ptxm_double_nan();

    const uint32_t x_bits = double_as_u64(x) >> 32;

    const uint32_t exp = (x_bits >> 20) & MASK_U32(11);
    const uint32_t frac = x_bits & MASK_U32(20);

    int x_log2 = (int)exp - 1023;

    uint32_t in_exp = 127u;

    if (x_log2 % 2 != 0)
    {
        x_log2 -= 1;
        in_exp += 1u;
    }

    const uint32_t in_bits = FP_FORMAT(0u, in_exp, frac << 3);

    const float in = u32_as_float(in_bits);
    const float rsqrt = ptxm_rsqrt_sm5x(in);

    const double r = ldexp(rsqrt, -x_log2 / 2);

    return u64_as_double(double_as_u64(r) & ~MASK_U64(32));
}
