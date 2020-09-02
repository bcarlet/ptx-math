#ifndef PTXS_PARAMS_H
#define PTXS_PARAMS_H

#include <stdint.h>

typedef struct ptxs_params
{
    const uint32_t (*table)[3];
    uint64_t bias;
} ptxs_params;

#endif
