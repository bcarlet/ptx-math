#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#include "cuda_util.hpp"
#include "ptx.hpp"

template<ptx_instruction I>
__global__
static void map(int n, float *x)
{
    int i, stride;

    GRID_STRIDE_LOOP(i, stride, n)
    {
        ptx_asm<I>::exec(x + i);
    }
}

int main()
{
    const int x_size = 1 << 20;
    float *x;

    CUDA_CHECK(cudaMallocManaged(&x, x_size * sizeof(float)));

    const float step = 0.000001f;

    for (int i = 0; i < x_size; i++)
    {
        x[i] = i * step;
    }

    const int block_dim = 256;
    const int grid_dim = (x_size + block_dim - 1) / block_dim;

    map<ptx_instruction::SIN_APPROX_F32><<<grid_dim, block_dim>>>(x_size, x);
    CUDA_CHECK(cudaPeekAtLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    float max_error = 0.0f;

    for (int i = 0; i < x_size; i++)
    {
        const float expected = sin(i * step);

        max_error = fmax(max_error, fabs(x[i] - expected));
    }

    printf("Max error: %.10f\n", max_error);

    CUDA_CHECK(cudaFree(x));

    return 0;
}
