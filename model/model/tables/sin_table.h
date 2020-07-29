#ifndef SIN_TABLE_H
#define SIN_TABLE_H

#include <stdint.h>

#define SIN_M 6
#define SIN_T 25
#define SIN_P 14
#define SIN_Q 10

#define SIN_C0_TERM_ALIGNMENT 31
#define SIN_C1_TERM_ALIGNMENT 19
#define SIN_C2_TERM_ALIGNMENT 0

#define SIN_SUM_WEIGHT 56

extern const uint32_t sin_table[64][3];

#endif
