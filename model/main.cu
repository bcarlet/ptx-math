#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>

using float_limits = std::numeric_limits<float>;

// sanity checks
static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");
static_assert(sizeof(float) == 4, "sizeof(float) != 4");
static_assert(!float_limits::traps, "float generates traps");

#include "cuda_util.hpp"
#include "ptx.hpp"

static constexpr uint32_t BATCH_SIZE = UINT32_C(1) << 20;
static constexpr uint32_t BATCH_COUNT = (UINT64_C(1) << 32) / BATCH_SIZE;

static constexpr int BLOCK_DIM = 1 << 8;
static constexpr int GRID_DIM = BATCH_SIZE / BLOCK_DIM;

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

static void initialize_batch(uint32_t batch, float *x)
{
    uint32_t val = batch * BATCH_SIZE;

    for (uint32_t i = 0; i < BATCH_SIZE; i++)
    {
        memcpy(x + i, &val, 4u);
        val++;
    }
}

static float compare_batch(uint32_t batch, const float *x, float (*f)(float))
{
    float max_error = 0.0f;
    uint32_t val = batch * BATCH_SIZE;

    for (uint32_t i = 0; i < BATCH_SIZE; i++)
    {
        float fval;
        memcpy(&fval, &val, 4u);

        max_error = fmax(max_error, fabs(x[i] - f(fval)));

        val++;
    }

    return max_error;
}

int main()
{
    float *x;
    CUDA_CHECK(cudaMallocManaged(&x, BATCH_SIZE * sizeof(float)));

    float max_error = 0.0f;

    for (uint32_t batch = 0; batch < BATCH_COUNT; batch++)
    {
        initialize_batch(batch, x);

        map<ptx_instruction::SIN_APPROX_F32><<<GRID_DIM, BLOCK_DIM>>>(BATCH_SIZE, x);
        CUDA_CHECK(cudaPeekAtLastError());

        CUDA_CHECK(cudaDeviceSynchronize());

        max_error = fmax(max_error, compare_batch(batch, x, sinf));
    }

    printf("Max error: %.10f\n", max_error);

    CUDA_CHECK(cudaFree(x));

    return 0;
}
