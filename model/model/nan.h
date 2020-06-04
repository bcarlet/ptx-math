#ifndef NAN_H
#define NAN_H

#include "fpmanip.h"

#define PTX_NAN UINT32_C(0x7fffffff)

inline float canonical_nan(void)
{
    return reinterpret_uint(PTX_NAN);
}

#endif
