#include "squarer.h"
#include "bitmask.h"

#include <immintrin.h>

#define TRUNC_COLS 19
#define PPM_ROWS 9

static uint32_t ppm_antidiagonal(uint32_t x)
{
    return _pdep_u32(x, UINT32_C(0x55555555));
}

static uint32_t ppm_row(uint32_t x, int i)
{
    if (!((x >> i) & 1u))
        return 0;

    x <<= i + 1;
    x &= ~MASK_U32(2 * i + 2);

    return x;
}

uint64_t ptxs_square_approx(uint32_t x)
{
    uint64_t error = ppm_antidiagonal(x) & MASK_U32(TRUNC_COLS);

    for (int i = 0; i < PPM_ROWS; i++)
    {
        error += ppm_row(x, i) & MASK_U32(TRUNC_COLS);
    }

    const uint64_t exact = (uint64_t)x * x;

    return exact - error;
}
