#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>

using float_limits = std::numeric_limits<float>;

// sanity checks
static_assert(CHAR_BIT == 8, "CHAR_BIT != 8");
static_assert(sizeof(float) == 4, "sizeof(float) != 4");
static_assert(!float_limits::traps, "float generates traps");

#include "model/model.h"
#include "util/progress.hpp"
#include "util/cuda.hpp"
#include "ptx/ptx.hpp"

static constexpr uint32_t BATCH_SIZE = UINT32_C(1) << 20;
static constexpr uint32_t BATCH_COUNT = (UINT64_C(1) << 32) / BATCH_SIZE;

static constexpr int BLOCK_DIM = 1 << 8;
static constexpr int GRID_DIM = BATCH_SIZE / BLOCK_DIM;

template<ptx_instruction I>
__global__
static void map(int n, float *x)
{
    GRID_STRIDE_LOOP(i, n)
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

static uint32_t compare_batch(uint32_t batch, const float *x, float (*f)(float))
{
    uint32_t num_exact = 0;
    uint32_t val = batch * BATCH_SIZE;

    for (uint32_t i = 0; i < BATCH_SIZE; i++)
    {
        float fval;
        memcpy(&fval, &val, 4u);

        const float cmp = f(fval);

        if (memcmp(x + i, &cmp, 4u) == 0)
            num_exact++;

        val++;
    }

    return num_exact;
}

int main()
{
    puts("Detecting devices...");
    print_devices();

    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    printf("Using device %d.\nRunning simulation...\n", device);

    float *x;
    CUDA_CHECK(cudaMallocManaged(&x, BATCH_SIZE * sizeof(float)));

    uint64_t num_exact = 0;

    for (uint32_t batch = 0; batch < BATCH_COUNT; batch++)
    {
        print_progress_bar((float)batch / BATCH_COUNT);
        initialize_batch(batch, x);

        map<ptx_instruction::RCP_APPROX_F32><<<GRID_DIM, BLOCK_DIM>>>(BATCH_SIZE, x);
        CUDA_CHECK(cudaPeekAtLastError());

        CUDA_CHECK(cudaDeviceSynchronize());

        num_exact += compare_batch(batch, x, model_rcp);
    }

    print_progress_bar(1.0f);
    putchar('\n');

    printf("Bit-exact: %llu\n", num_exact);

    CUDA_CHECK(cudaFree(x));

    return 0;
}
