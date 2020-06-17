#ifndef TUNING_H
#define TUNING_H

#include "params.h"

#ifdef __cplusplus
extern "C" {
#endif

float parameterized_rcp(float x, const m_params *params);

#ifdef __cplusplus
}   // extern "C"
#endif

#endif
