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

    return (float)(1.0 / sqrt((double)x));
}
