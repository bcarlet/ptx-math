#include "model.h"
#include "nan.h"

#include <math.h>

float model_cos(float x)
{
    switch (fpclassify(x))
    {
    case FP_NAN:
        return canonical_nan();
    case FP_INFINITE:
        return canonical_nan();
    case FP_ZERO:
        return 1.0f;
#ifdef PTX_FTZ
    case FP_SUBNORMAL:
        return 1.0f;
#endif
    }

    return cosf(x);
}
