#ifndef PTXS_TUNING_H
#define PTXS_TUNING_H

#include "params.h"

#ifdef __cplusplus
extern "C" {
#endif

float ptxs_param_rcp(float x, const ptxs_params *params);

float ptxs_param_sqrt(float x, const ptxs_params *params);

float ptxs_param_rsqrt(float x, const ptxs_params *params);

float ptxs_param_sin(float x, const ptxs_params *params);

float ptxs_param_lg2(float x, const ptxs_params *params);

float ptxs_param_ex2(float x, const ptxs_params *params);

#ifdef __cplusplus
}   // extern "C"
#endif

#endif
