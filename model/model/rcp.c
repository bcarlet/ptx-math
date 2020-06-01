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

    // x = fraction * 2^exp
    // fraction in (-1, -0.5] U [0.5, 1)
    int exp;
    float fraction = frexpf(x, &exp);

    // scale fraction to (-2, -1] U [1, 2)
    exp -= 1;
    fraction *= 2;

    float r = 1.0f / fraction;

    return ldexpf(r, -exp);
}
