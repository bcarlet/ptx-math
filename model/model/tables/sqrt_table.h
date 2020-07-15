#ifndef SQRT_TABLE_H
#define SQRT_TABLE_H

#include <stdint.h>

#define SQRT_M 6
#define SQRT_T 25
#define SQRT_P 17
#define SQRT_Q 12

#define SQRT_C0_TERM_ALIGNMENT 33
#define SQRT_C1_TERM_ALIGNMENT 18
#define SQRT_C2_TERM_ALIGNMENT 0

#define SQRT_SUM_WEIGHT 58

extern const uint32_t sqrt_table[128][3];

#endif
