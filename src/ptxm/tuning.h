#ifndef PTXM_TUNING_H
#define PTXM_TUNING_H

#include "params.h"

#ifdef __cplusplus
extern "C" {
#endif

float ptxm_rcp_sm5x_internal(float x, const ptxm_params *params);
float ptxm_sqrt_sm6x_internal(float x, const ptxm_params *params);
float ptxm_rsqrt_sm5x_internal(float x, const ptxm_params *params);
float ptxm_sin_sm5x_internal(float x, const ptxm_params *params);
float ptxm_lg2_sm5x_internal(float x, const ptxm_params *params);
float ptxm_ex2_sm5x_internal(float x, const ptxm_params *params);

#ifdef __cplusplus
}   // extern "C"
#endif

#endif
