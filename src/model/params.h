#ifndef PARAMS_H
#define PARAMS_H

#include <stdint.h>

typedef struct m_params
{
    const uint32_t (*table)[3];
    uint64_t bias;
    int truncation;
} m_params;

#endif
