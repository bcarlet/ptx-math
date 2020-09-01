#ifndef NAN_H
#define NAN_H

#include "bitcast.h"

#define PTX_CANONICAL_NAN UINT32_C(0x7fffffff)

inline float ptxs_nan(void)
{
    return u32_as_float(PTX_CANONICAL_NAN);
}

#endif
