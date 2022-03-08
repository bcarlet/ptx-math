#ifndef PTXM_COMMON_NAN_H
#define PTXM_COMMON_NAN_H

#include "bitcast.h"

#define PTX_CANONICAL_NAN           UINT32_C(0x7fffffff)
#define PTX_CANONICAL_DOUBLE_NAN    UINT64_C(0x7fffffff00000000)

inline float ptxm_nan(void)
{
    return u32_as_float(PTX_CANONICAL_NAN);
}

inline double ptxm_double_nan(void)
{
    return u64_as_double(PTX_CANONICAL_DOUBLE_NAN);
}

#endif
