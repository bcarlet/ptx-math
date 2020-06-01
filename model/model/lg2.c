#include "model.h"
#include "nan.h"

#include <math.h>

float model_lg2(float x)
{
    switch (fpclassify(x))
    {
    case FP_NAN:
        return canonical_nan();
    case FP_INFINITE:
        if (signbit(x)) return canonical_nan();
        else return INFINITY;
    case FP_ZERO:
        return -INFINITY;
#ifdef PTX_FTZ
    case FP_SUBNORMAL:
        return -INFINITY;
#endif
    }

    if (signbit(x)) return canonical_nan();

    return log2f(x);
}
