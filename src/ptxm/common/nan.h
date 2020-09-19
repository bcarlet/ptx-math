#ifndef PTXM_COMMON_NAN_H
#define PTXM_COMMON_NAN_H

#include "bitcast.h"

#define PTX_CANONICAL_NAN UINT32_C(0x7fffffff)

inline float ptxm_nan(void)
{
    return u32_as_float(PTX_CANONICAL_NAN);
}

#endif
