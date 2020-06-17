#ifndef RCP_TABLE_H
#define RCP_TABLE_H

#include <stdint.h>

#define RCP_M 7
#define RCP_T 26
#define RCP_P 16
#define RCP_Q 10

#define RCP_C0_TERM_ALIGNMENT 30
#define RCP_C1_TERM_ALIGNMENT 17
#define RCP_C2_TERM_ALIGNMENT 0

#define RCP_SUM_WEIGHT 56

extern const uint32_t rcp_table[128][3];

#endif
