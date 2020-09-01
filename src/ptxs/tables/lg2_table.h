#ifndef LG2_TABLE_H
#define LG2_TABLE_H

#include <stdint.h>

#define LG2_M 6
#define LG2_T 26
#define LG2_P 15
#define LG2_Q 10

#define LG2_C0_TERM_ALIGNMENT 30
#define LG2_C1_TERM_ALIGNMENT 18
#define LG2_C2_TERM_ALIGNMENT 0

#define LG2_SUM_WEIGHT 56

extern const uint32_t ptxs_lg2_table[64][3];

#endif
