#ifndef RSQRT_TABLE_H
#define RSQRT_TABLE_H

#include <stdint.h>

#define RSQRT_M 6
#define RSQRT_T 26
#define RSQRT_P 16
#define RSQRT_Q 10

#define RSQRT_C0_TERM_ALIGNMENT 30
#define RSQRT_C1_TERM_ALIGNMENT 17
#define RSQRT_C2_TERM_ALIGNMENT 0

#define RSQRT_SUM_WEIGHT 56

extern const uint32_t rsqrt_table[128][3];

#endif
