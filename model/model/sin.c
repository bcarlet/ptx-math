#include "model.h"
#include "common/nan.h"

#include <math.h>

float model_sin(float x)
{
    switch (fpclassify(x))
    {
    case FP_NAN:
        return canonical_nan();
    case FP_INFINITE:
        return canonical_nan();
    case FP_ZERO:
        return copysignf(0.0f, x);
#ifdef PTX_FTZ
    case FP_SUBNORMAL:
        return copysignf(0.0f, x);
#endif
    }

    return sinf(x);
}
