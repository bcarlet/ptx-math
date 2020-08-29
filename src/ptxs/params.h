#ifndef PARAMS_H
#define PARAMS_H

#include <stdint.h>

typedef struct ptxs_params
{
    const uint32_t (*table)[3];
    uint64_t bias;
} ptxs_params;

#endif
