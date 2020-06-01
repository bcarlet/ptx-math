#include "model.h"
#include "nan.h"

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

    // x = frac * 2^exp
    // frac in (-2, -1] U [1, 2)
    const int exp = ilogbf(x);
    const float frac = ldexpf(x, -exp);

    float r = 1.0f / frac;

    return ldexpf(r, -exp);
}
