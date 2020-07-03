#include "squarer.h"
#include "bitmask.h"

#include <immintrin.h>

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

uint64_t square_approx(uint32_t x, int t_cols)
{
    const int rows = (t_cols + 1) / 2 - 1;
    const uint32_t mask = MASK_U32(t_cols);

    uint64_t error = ppm_antidiagonal(x) & mask;

    for (int i = 0; i < rows; i++)
    {
        error += ppm_row(x, i) & mask;
    }

    const uint64_t exact = (uint64_t)x * x;

    return exact - error;
}
