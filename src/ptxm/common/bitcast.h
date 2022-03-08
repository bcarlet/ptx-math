#ifndef PTXM_COMMON_BITCAST_H
#define PTXM_COMMON_BITCAST_H

#include <assert.h>
#include <stdint.h>
#include <string.h>

#define MEMCPY_CHECKED(dst, src) \
    do { \
        static_assert(sizeof(*(dst)) == sizeof(*(src)), "size mismatch"); \
        memcpy(dst, src, sizeof(*(dst))); \
    } while (0)

inline uint32_t float_as_u32(float x)
{
    uint32_t r;
    MEMCPY_CHECKED(&r, &x);

    return r;
}

inline uint64_t double_as_u64(double x)
{
    uint64_t r;
    MEMCPY_CHECKED(&r, &x);

    return r;
}

inline float u32_as_float(uint32_t x)
{
    float r;
    MEMCPY_CHECKED(&r, &x);

    return r;
}

inline double u64_as_double(uint64_t x)
{
    double r;
    MEMCPY_CHECKED(&r, &x);

    return r;
}

#endif
