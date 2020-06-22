#include "model.h"
#include "common/nan.h"

#include <math.h>

float model_ex2(float x)
{
    switch (fpclassify(x))
    {
    case FP_NAN:
        return canonical_nan();
    case FP_INFINITE:
        return signbit(x) ? 0.0f : INFINITY;
    case FP_ZERO:
        return 1.0f;
#ifdef PTX_FTZ
    case FP_SUBNORMAL:
        return 1.0f;
#endif
    }

    return exp2f(x);
}
