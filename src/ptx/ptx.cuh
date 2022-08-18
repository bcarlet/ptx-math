#ifndef PTX_PTX_CUH
#define PTX_PTX_CUH

#include <cuda_runtime.h>

namespace ptx
{

enum class opcode
{
    RCP_APPROX_F32,
    SQRT_APPROX_F32,
    RSQRT_APPROX_F32,
    SIN_APPROX_F32,
    COS_APPROX_F32,
    LG2_APPROX_F32,
    EX2_APPROX_F32
};

template<opcode I>
struct asm_traits {};

template<>
struct asm_traits<opcode::RCP_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("rcp.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct asm_traits<opcode::SQRT_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("sqrt.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct asm_traits<opcode::RSQRT_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("rsqrt.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct asm_traits<opcode::SIN_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("sin.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct asm_traits<opcode::COS_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("cos.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct asm_traits<opcode::LG2_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("lg2.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct asm_traits<opcode::EX2_APPROX_F32>
{
    __device__ __forceinline__ static void exec(float *x)
    {
        asm("ex2.approx.f32 %0, %0;" : "+f"(*x));
    }
};

}   // namespace ptx

#endif
