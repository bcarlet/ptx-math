#ifndef PTXM_PARAMS_H
#define PTXM_PARAMS_H

#include <stdint.h>

typedef struct ptxm_params
{
    const uint32_t (*table)[3];
    uint64_t bias;
} ptxm_params;

#endif
