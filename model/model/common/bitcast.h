#ifndef BITCAST_H
#define BITCAST_H

#include <stdint.h>
#include <string.h>

#include "fpassert.h"

inline uint32_t float_as_u32(float x)
{
    uint32_t r;
    memcpy(&r, &x, 4u);

    return r;
}

inline float u32_as_float(uint32_t x)
{
    float r;
    memcpy(&r, &x, 4u);

    return r;
}

#endif
