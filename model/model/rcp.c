#include "model.h"
#include "fpmanip.h"
#include "nan.h"

#include <math.h>

#define RCP_M 7
#define RCP_T 26
#define RCP_P 16
#define RCP_Q 10

static const uint32_t table[128][3];

float model_rcp(const float x)
{
    switch (fpclassify(x))
    {
    case FP_NAN:
        return canonical_nan();
    case FP_INFINITE:
        return copysignf(0.0f, x);
    case FP_ZERO:
        return copysignf(INFINITY, x);
#ifdef PTX_FTZ
    case FP_SUBNORMAL:
        return copysignf(INFINITY, x);
#else
    case FP_SUBNORMAL:
        return 1.0f / x;    // can't handle subnormals explicitly yet
#endif
    }

    const uint32_t x_bits = reinterpret_float(x);

    const uint32_t xh = UPPER_SIGNIFICAND(x_bits, RCP_M);
    const uint32_t xl = LOWER_SIGNIFICAND(x_bits, RCP_M);

    const uint32_t *const c = table[xh];

    const uint32_t c1_term = c[1] * xl;
    const uint64_t c2_term = (uint64_t)c[2] * (xl * xl);

    const uint64_t c0_aligned = (uint64_t)c[0] << 30;
    const uint64_t c1_term_aligned = (uint64_t)c1_term << 17;

    uint64_t sum = c0_aligned - c1_term_aligned + c2_term;

    sum += UINT64_C(1) << (56 - 24);

    uint32_t r_exp = 127u;

    if (!(sum >> 56))
    {
        r_exp -= 1;
        sum <<= 1;
    }

    const uint32_t r_sig = (sum >> (56 - 23)) & MASK_U32(23);
    const uint32_t r_sign = signbit(x) ? 1u : 0u;

    const uint32_t r_bits = (r_sign << 31) | (r_exp << 23) | r_sig;

    const float r = reinterpret_uint(r_bits);

    return ldexpf(r, -ilogbf(x));
}

const uint32_t table[128][3] =
{
    {0x3ffffff, 0xfffe, 0x3f5}, {0x3f80fdf, 0xfc0a, 0x3dd}, {0x3f03f03, 0xf82d, 0x3c6}, {0x3e88cb3, 0xf467, 0x3b1},
    {0x3e0f83d, 0xf0b6, 0x39b}, {0x3d980f6, 0xed1b, 0x386}, {0x3d22635, 0xe995, 0x373}, {0x3cae758, 0xe622, 0x35f},
    {0x3c3c3c3, 0xe2c3, 0x34d}, {0x3bcbadc, 0xdf77, 0x33b}, {0x3b5cc0e, 0xdc3d, 0x329}, {0x3aef6ca, 0xd914, 0x317},
    {0x3a83a83, 0xd5fd, 0x306}, {0x3a196b1, 0xd2f7, 0x2f6}, {0x39b0ad0, 0xd001, 0x2e6}, {0x3949660, 0xcd1b, 0x2d7},
    {0x38e38e3, 0xca44, 0x2c7}, {0x387f1e0, 0xc77c, 0x2b8}, {0x381c0e0, 0xc4c3, 0x2aa}, {0x37ba571, 0xc218, 0x29d},
    {0x3759f22, 0xbf7b, 0x290}, {0x36fad87, 0xbceb, 0x282}, {0x369d036, 0xba69, 0x277}, {0x36406c8, 0xb7f3, 0x26a},
    {0x35e50d7, 0xb589, 0x25d}, {0x358ae03, 0xb32c, 0x252}, {0x3531dec, 0xb0da, 0x246}, {0x34da034, 0xae94, 0x23c},
    {0x3483483, 0xac59, 0x231}, {0x342da7f, 0xaa28, 0x225}, {0x33d91d2, 0xa803, 0x21c}, {0x3385a29, 0xa5e7, 0x211},
    {0x3333333, 0xa3d6, 0x207}, {0x32e1c9f, 0xa1cf, 0x1fe}, {0x329161f, 0x9fd1, 0x1f5}, {0x3241f69, 0x9ddc, 0x1ea},
    {0x31f3832, 0x9bf1, 0x1e2}, {0x31a6031, 0x9a0f, 0x1da}, {0x3159722, 0x9835, 0x1d1}, {0x310dcbe, 0x9664, 0x1c9},
    {0x30c30c3, 0x949b, 0x1c1}, {0x30792ef, 0x92da, 0x1b9}, {0x3030303, 0x9121, 0x1b1}, {0x2fe80bf, 0x8f70, 0x1aa},
    {0x2fa0be8, 0x8dc6, 0x1a2}, {0x2f5a441, 0x8c24, 0x19c}, {0x2f14990, 0x8a88, 0x193}, {0x2ecfb9c, 0x88f4, 0x18d},
    {0x2e8ba2e, 0x8767, 0x186}, {0x2e48510, 0x85e0, 0x17f}, {0x2e05c0b, 0x8460, 0x179}, {0x2dc3eed, 0x82e7, 0x173},
    {0x2d82d83, 0x8174, 0x16e}, {0x2d4279a, 0x8006, 0x166}, {0x2d02d03, 0x7e9f, 0x161}, {0x2cc3d8d, 0x7d3e, 0x15c},
    {0x2c8590b, 0x7be2, 0x155}, {0x2c47f4f, 0x7a8d, 0x151}, {0x2c0b02c, 0x793c, 0x14b}, {0x2bceb77, 0x77f1, 0x146},
    {0x2b93105, 0x76ab, 0x140}, {0x2b580ad, 0x756b, 0x13c}, {0x2b1da46, 0x742f, 0x136}, {0x2ae3da7, 0x72f8, 0x131},
    {0x2aaaaaa, 0x71c7, 0x12e}, {0x2a72129, 0x709a, 0x129}, {0x2a3a0fd, 0x6f71, 0x123}, {0x2a02a02, 0x6e4d, 0x11f},
    {0x29cbc15, 0x6d2e, 0x11b}, {0x2995711, 0x6c13, 0x117}, {0x295fad4, 0x6afc, 0x112}, {0x292a73c, 0x69ea, 0x10f},
    {0x28f5c29, 0x68db, 0x10a}, {0x28c1979, 0x67d1, 0x107}, {0x288df0d, 0x66ca, 0x102}, {0x285acc5, 0x65c8, 0x100},
    {0x2828282, 0x64c9, 0x0fb}, {0x27f6028, 0x63ce, 0x0f8}, {0x27c4597, 0x62d6, 0x0f3}, {0x27932b4, 0x61e2, 0x0f0},
    {0x2762762, 0x60f2, 0x0ed}, {0x2732385, 0x6005, 0x0e9}, {0x2702702, 0x5f1c, 0x0e7}, {0x26d31be, 0x5e35, 0x0e2},
    {0x26a439f, 0x5d52, 0x0df}, {0x2675c8b, 0x5c73, 0x0de}, {0x2647c69, 0x5b96, 0x0da}, {0x261a320, 0x5abc, 0x0d6},
    {0x25ed098, 0x59e6, 0x0d4}, {0x25c04b8, 0x5912, 0x0d0}, {0x2593f6a, 0x5841, 0x0cd}, {0x2568096, 0x5774, 0x0cc},
    {0x253c825, 0x56a8, 0x0c7}, {0x2511602, 0x55e0, 0x0c5}, {0x24e6a17, 0x551b, 0x0c4}, {0x24bc44e, 0x5458, 0x0c1},
    {0x2492492, 0x5397, 0x0bd}, {0x2468acf, 0x52d9, 0x0ba}, {0x243f6f0, 0x521e, 0x0b8}, {0x24168e1, 0x5165, 0x0b6},
    {0x23ee090, 0x50af, 0x0b4}, {0x23c5de7, 0x4ffb, 0x0b2}, {0x239e0d6, 0x4f49, 0x0af}, {0x2376948, 0x4e9a, 0x0ad},
    {0x234f72c, 0x4ded, 0x0ab}, {0x2328a70, 0x4d42, 0x0a9}, {0x2302302, 0x4c99, 0x0a6}, {0x22dc0d1, 0x4bf3, 0x0a5},
    {0x22b63cc, 0x4b4e, 0x0a1}, {0x2290be2, 0x4aac, 0x0a0}, {0x226b902, 0x4a0c, 0x09f}, {0x2246b1d, 0x496d, 0x09b},
    {0x2222222, 0x48d1, 0x09a}, {0x21fde02, 0x4837, 0x099}, {0x21d9ead, 0x479e, 0x096}, {0x21b6415, 0x4708, 0x095},
    {0x2192e2a, 0x4673, 0x093}, {0x216fcdd, 0x45e0, 0x091}, {0x214d021, 0x454f, 0x08f}, {0x212a7e7, 0x44c0, 0x08f},
    {0x2108421, 0x4432, 0x08c}, {0x20e64c1, 0x43a6, 0x08a}, {0x20c49ba, 0x431c, 0x089}, {0x20a32ff, 0x4293, 0x087},
    {0x2082082, 0x420c, 0x085}, {0x2061237, 0x4187, 0x084}, {0x2040810, 0x4103, 0x083}, {0x2020202, 0x4081, 0x082}
};
