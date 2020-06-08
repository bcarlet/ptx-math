#include "model.h"
#include "common/nan.h"

#include <math.h>

float model_sqrt(float x)
{
    switch (fpclassify(x))
    {
    case FP_NAN:
        return canonical_nan();
    case FP_INFINITE:
        if (signbit(x)) return canonical_nan();
        else return INFINITY;
    case FP_ZERO:
        return copysignf(0.0f, x);
#ifdef PTX_FTZ
    case FP_SUBNORMAL:
        return copysignf(0.0f, x);
#endif
    }

    if (signbit(x)) return canonical_nan();

    return sqrtf(x);
}
