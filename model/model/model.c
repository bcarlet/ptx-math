#include "model.h"

#include <math.h>

float model_rcp(float x)
{
    return 1.0f / x;
}

float model_sqrt(float x)
{
    return sqrtf(x);
}

float model_rsqrt(float x)
{
    return (float)(1.0 / sqrt((double)x));
}

float model_sin(float x)
{
    return sinf(x);
}

float model_cos(float x)
{
    return cosf(x);
}

float model_lg2(float x)
{
    return log2f(x);
}

float model_ex2(float x)
{
    return exp2f(x);
}
