#include "model.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static inline float canonical_nan(void)
{
    const uint32_t inan = UINT32_C(0x7fffffff);

    float fnan;
    memcpy(&fnan, &inan, 4u);

    return fnan;
}

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
#endif
    }

    return 1.0f / x;
}

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

float model_rsqrt(float x)
{
    switch (fpclassify(x))
    {
    case FP_NAN:
        return canonical_nan();
    case FP_INFINITE:
        if (signbit(x)) return canonical_nan();
        else return 0.0f;
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

float model_ex2(float x)
{
    switch (fpclassify(x))
    {
    case FP_NAN:
        return canonical_nan();
    case FP_INFINITE:
        if (signbit(x)) return 0.0f;
        else return INFINITY;
    case FP_ZERO:
        return 1.0f;
#ifdef PTX_FTZ
    case FP_SUBNORMAL:
        return 1.0f;
#endif
    }

    return exp2f(x);
}
