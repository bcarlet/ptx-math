#include "model.h"
#include "common/nan.h"

#include <math.h>

float model_rsqrt(float x)
{
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
#endif
    }

    if (signbit(x)) return canonical_nan();

    int x_log2 = ilogbf(x);
    float frac = ldexpf(x, -x_log2);

    if (x_log2 % 2 != 0)
    {
        x_log2 -= 1;
        frac *= 2.0f;
    }

    const float r = (float)(1.0 / sqrt((double)frac));

    return ldexpf(r, -x_log2 / 2);
}
