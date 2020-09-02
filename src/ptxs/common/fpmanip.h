#ifndef PTXS_COMMON_FPMANIP_H
#define PTXS_COMMON_FPMANIP_H

#include "bitmask.h"

#define UPPER_SIGNIFICAND(x, m) (((x) >> (23 - (m))) & MASK_U32(m))
#define LOWER_SIGNIFICAND(x, m) ((x) & MASK_U32(23 - (m)))

#define FP_FORMAT(sign, exp, frac) (((sign) << 31) | ((exp) << 23) | (frac))

#endif
