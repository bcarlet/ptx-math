#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#include "cuda_check.h"

__global__
static void sine(int n, float *x)
{
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        asm("sin.approx.f32 %0, %0;" : "+f"(x[i]));
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

    sine<<<grid_dim, block_dim>>>(x_size, x);
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
