#ifndef NAN_H
#define NAN_H

#include <stdint.h>
#include <string.h>

inline float canonical_nan(void)
{
    const uint32_t inan = UINT32_C(0x7fffffff);

    float fnan;
    memcpy(&fnan, &inan, 4u);

    return fnan;
}

#endif
