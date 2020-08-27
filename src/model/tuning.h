#ifndef TUNING_H
#define TUNING_H

#include "params.h"

#ifdef __cplusplus
extern "C" {
#endif

float parameterized_rcp(float x, const m_params *params);

float parameterized_sqrt(float x, const m_params *params);

float parameterized_rsqrt(float x, const m_params *params);

float parameterized_sin(float x, const m_params *params);

float parameterized_lg2(float x, const m_params *params);

float parameterized_ex2(float x, const m_params *params);

#ifdef __cplusplus
}   // extern "C"
#endif

#endif
