#include "rro_sincos.h"

#include "../common/bitcast.h"
#include "../common/fpmanip.h"

#include <math.h>

#define SINCOS_PI_2_RCP 0x1.45f306p-1f
#define SINCOS_PI_2     0x1.921fb6546df20p0
#define SINCOS_Q0_MAX   0x1.921fb6p0f

static int min(int a, int b);
static int max(int a, int b);

static uint32_t lerp_sincos(double x)
{
    x *= SINCOS_PI_2_RCP;

    if (x == 0.0) return 0u;

    uint32_t x_bits = double_as_u64(x) >> 29;

    x_bits |= UINT32_C(1) << 23;
    x_bits &= MASK_U32(24);

    x_bits >>= min(-ilogb(x), 24);

    return x_bits;
}

uint32_t ptxs_rro_sincos(float x)
{
    const uint32_t sign = signbit(x) ? 1u : 0u;

    switch (fpclassify(x))
    {
    case FP_NAN:
        return FP_FORMAT(sign, 0x80u, 0u);
    case FP_INFINITE:
        return FP_FORMAT(sign, 0x81u, 0u);
    case FP_ZERO:
    case FP_SUBNORMAL:
        return FP_FORMAT(sign, 0u, 0u);
    }

    x = fabsf(x);

    if (x >= 0x1p33f) return FP_FORMAT(sign, 0u, 0u);

    const uint64_t k = (double)x * SINCOS_PI_2_RCP;

    const double reduced = x - k * SINCOS_PI_2;     // may be contracted
    const double clamped = fmin(fmax(reduced, 0.0), SINCOS_Q0_MAX);

    uint32_t q = k & MASK_U64(2);
    uint32_t r = lerp_sincos(clamped);

    const int x_log2 = ilogbf(x);

    q &= ~MASK_U32(max(x_log2 - 31, 0));
    r &= ~MASK_U32(max(x_log2 - 8, 0));

    return FP_FORMAT(sign, q, r);
}

int min(int a, int b)
{
    return (a < b) ? a : b;
}

int max(int a, int b)
{
    return (a > b) ? a : b;
}
