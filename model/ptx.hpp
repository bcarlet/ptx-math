#ifndef PTX_HPP
#define PTX_HPP

#include <cuda_runtime.h>

__device__
inline void rcp_approx_f32(float *x)
{
    asm("rcp.approx.f32 %0, %0;" : "+f"(*x));
}

__device__
inline void sqrt_approx_f32(float *x)
{
    asm("sqrt.approx.f32 %0, %0;" : "+f"(*x));
}

__device__
inline void rsqrt_approx_f32(float *x)
{
    asm("rsqrt.approx.f32 %0, %0;" : "+f"(*x));
}

__device__
inline void sin_approx_f32(float *x)
{
    asm("sin.approx.f32 %0, %0;" : "+f"(*x));
}

__device__
inline void cos_approx_f32(float *x)
{
    asm("cos.approx.f32 %0, %0;" : "+f"(*x));
}

#endif
