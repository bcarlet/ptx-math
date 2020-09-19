#ifndef PTXM_COMMON_UTIL_H
#define PTXM_COMMON_UTIL_H

#include "bitmask.h"

// Extract upper bits from the lower bits of x.
#define EXTRACT_BITS(x, count, from) \
    (((x) >> ((from) - (count))) & MASK_U32(count))

#define UPPER_SIGNIFICAND(x, m) \
    EXTRACT_BITS(x, m, 23)

#define LOWER_SIGNIFICAND(x, m) \
    ((x) & MASK_U32(23 - (m)))

#define FP_FORMAT(sign, ex, frac) \
    (((sign) << 31) | ((ex) << 23) | (frac))

#endif
