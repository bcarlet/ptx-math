#ifndef FPMANIP_H
#define FPMANIP_H

#include <stdint.h>
#include <string.h>

#include "bitmask.h"
#include "fpassert.h"

#define UPPER_SIGNIFICAND(x, m) (((x) >> (23 - (m))) & MASK_U32(m))

#define LOWER_SIGNIFICAND(x, m) ((x) & MASK_U32(23 - (m)))

inline uint32_t reinterpret_float(float x)
{
    uint32_t i;
    memcpy(&i, &x, 4u);

    return i;
}

inline float reinterpret_uint(uint32_t x)
{
    float f;
    memcpy(&f, &x, 4u);

    return f;
}

#endif
