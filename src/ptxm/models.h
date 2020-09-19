#ifndef PTXM_MODELS_H
#define PTXM_MODELS_H

#ifdef __cplusplus
extern "C" {
#endif

float ptxm_rcp_sm5x(float x);
float ptxm_sqrt_sm5x(float x);
float ptxm_sqrt_sm6x(float x);
float ptxm_rsqrt_sm5x(float x);
float ptxm_sin_sm5x(float x);
float ptxm_cos_sm5x(float x);
float ptxm_lg2_sm5x(float x);
float ptxm_ex2_sm5x(float x);

#ifdef __cplusplus
}   // extern "C"
#endif

#endif
