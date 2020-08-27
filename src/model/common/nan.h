#ifndef NAN_H
#define NAN_H

#include "bitcast.h"

#define PTX_NAN UINT32_C(0x7fffffff)

inline float canonical_nan(void)
{
    return u32_as_float(PTX_NAN);
}

#endif
