#ifndef PTX_PTX_CUH
#define PTX_PTX_CUH

#include <cuda_runtime.h>

namespace ptx
{

enum class opcode
{
    RCP_APPROX_F32,
    RCP_APPROX_FTZ_F64,
    SQRT_APPROX_F32,
    RSQRT_APPROX_F32,
    RSQRT_APPROX_FTZ_F64,
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
    using operand_type = float;

    __device__ __forceinline__
    static void exec(operand_type *x)
    {
        asm("rcp.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct asm_traits<opcode::RCP_APPROX_FTZ_F64>
{
    using operand_type = double;

    __device__ __forceinline__
    static void exec(operand_type *x)
    {
        asm("rcp.approx.ftz.f64 %0, %0;" : "+d"(*x));
    }
};

template<>
struct asm_traits<opcode::SQRT_APPROX_F32>
{
    using operand_type = float;

    __device__ __forceinline__
    static void exec(operand_type *x)
    {
        asm("sqrt.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct asm_traits<opcode::RSQRT_APPROX_F32>
{
    using operand_type = float;

    __device__ __forceinline__
    static void exec(operand_type *x)
    {
        asm("rsqrt.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct asm_traits<opcode::RSQRT_APPROX_FTZ_F64>
{
    using operand_type = double;

    __device__ __forceinline__
    static void exec(operand_type *x)
    {
        asm("rsqrt.approx.ftz.f64 %0, %0;" : "+d"(*x));
    }
};

template<>
struct asm_traits<opcode::SIN_APPROX_F32>
{
    using operand_type = float;

    __device__ __forceinline__
    static void exec(operand_type *x)
    {
        asm("sin.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct asm_traits<opcode::COS_APPROX_F32>
{
    using operand_type = float;

    __device__ __forceinline__
    static void exec(operand_type *x)
    {
        asm("cos.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct asm_traits<opcode::LG2_APPROX_F32>
{
    using operand_type = float;

    __device__ __forceinline__
    static void exec(operand_type *x)
    {
        asm("lg2.approx.f32 %0, %0;" : "+f"(*x));
    }
};

template<>
struct asm_traits<opcode::EX2_APPROX_F32>
{
    using operand_type = float;

    __device__ __forceinline__
    static void exec(operand_type *x)
    {
        asm("ex2.approx.f32 %0, %0;" : "+f"(*x));
    }
};

}   // namespace ptx

#endif
