#include "models.h"

#include "common/bitcast.h"
#include "common/nan.h"
#include "common/util.h"

#include <math.h>

double ptxm_rcp_ftz_f64_sm5x(double x)
{
    x = u64_as_double(double_as_u64(x) & ~MASK_U64(32));

    switch (fpclassify(x))
    {
    case FP_NAN:
        return ptxm_double_nan();
    case FP_INFINITE:
        return copysign(0.0, x);
    case FP_ZERO:
    case FP_SUBNORMAL:
        return copysign(INFINITY, x);
    }

    const uint32_t x_bits = double_as_u64(x) >> 32;

    const uint32_t sign = x_bits >> 31;
    const uint32_t exp = (x_bits >> 20) & MASK_U32(11);
    const uint32_t frac = x_bits & MASK_U32(20);

    const int x_log2 = (int)exp - 1023;

    if (x_log2 >= 1022 && (x_log2 > 1022 || frac != 0u))
        return copysign(0.0, x);

    const uint32_t in_bits = FP_FORMAT(sign, 127u, frac << 3);

    const float in = u32_as_float(in_bits);
    const float rcp = ptxm_rcp_sm5x(in);

    const double r = ldexp(rcp, -x_log2);

    return u64_as_double(double_as_u64(r) & ~MASK_U64(32));
}
