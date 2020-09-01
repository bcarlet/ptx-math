#ifndef PTX_CUH
#define PTX_CUH

#include <cuda_runtime.h>

enum class ptx_instruction
{
    RCP_APPROX_F32,
    SQRT_APPROX_F32,
    RSQRT_APPROX_F32,
    SIN_APPROX_F32,
    COS_APPROX_F32,
    LG2_APPROX_F32,
    EX2_APPROX_F32
};

template<ptx_instruction I>
struct ptx_asm
{
    __device__ __forceinline__ static void exec(float *x);
};

template<>
struct ptx_asm<ptx_instruction::RCP_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("rcp.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct ptx_asm<ptx_instruction::SQRT_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("sqrt.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct ptx_asm<ptx_instruction::RSQRT_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("rsqrt.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct ptx_asm<ptx_instruction::SIN_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("sin.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct ptx_asm<ptx_instruction::COS_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("cos.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct ptx_asm<ptx_instruction::LG2_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("lg2.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct ptx_asm<ptx_instruction::EX2_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("ex2.approx.f32 %0, %0;" : "+f"(*x));
    }
};

#endif
