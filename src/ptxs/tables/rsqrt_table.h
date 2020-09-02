#ifndef PTXS_TABLES_RSQRT_TABLE_H
#define PTXS_TABLES_RSQRT_TABLE_H

#include <stdint.h>

#define RSQRT_M 6
#define RSQRT_T 26
#define RSQRT_P 17
#define RSQRT_Q 11

#define RSQRT_C0_TERM_ALIGNMENT 31
#define RSQRT_C1_TERM_ALIGNMENT 17
#define RSQRT_C2_TERM_ALIGNMENT 0

#define RSQRT_SUM_WEIGHT 57

extern const uint32_t ptxs_rsqrt_table[128][3];

#endif
