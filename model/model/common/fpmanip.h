#ifndef FPMANIP_H
#define FPMANIP_H

#include <stdint.h>
#include <string.h>

#include "bitmask.h"
#include "fpassert.h"

#define UPPER_SIGNIFICAND(x, m) (((x) >> (23 - (m))) & MASK_U32(m))

#define LOWER_SIGNIFICAND(x, m) ((x) & MASK_U32(23 - (m)))

#define FP_FORMAT(sign, exp, frac) (((sign) << 31) | ((exp) << 23) | (frac))

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
