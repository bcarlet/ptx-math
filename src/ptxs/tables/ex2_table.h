#ifndef EX2_TABLE_H
#define EX2_TABLE_H

#include <stdint.h>

#define EX2_M 6
#define EX2_T 25
#define EX2_P 15
#define EX2_Q 11

#define EX2_C0_TERM_ALIGNMENT 32
#define EX2_C1_TERM_ALIGNMENT 19
#define EX2_C2_TERM_ALIGNMENT 0

#define EX2_SUM_WEIGHT 57

extern const uint32_t ex2_table[64][3];

#endif
