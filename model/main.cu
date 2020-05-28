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

#include "model/model.h"
#include "util/progress.hpp"
#include "util/tally.hpp"
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

struct comp_stats
{
    void accumulate(float a, float b)
    {
        if (isfinite(a) && isfinite(b))
            error.accumulate(fabs((double)b - a));

        if (memcmp(&a, &b, sizeof(float)) == 0)
            num_exact++;
    }

    tally<double> error;
    unsigned long long num_exact = 0u;
};

static void compare_batch(uint32_t batch, const float *x, float (*f)(float), comp_stats &stats)
{
    uint32_t val = batch * BATCH_SIZE;

    for (uint32_t i = 0; i < BATCH_SIZE; i++)
    {
        float fval;
        memcpy(&fval, &val, 4u);

        stats.accumulate(x[i], f(fval));

        val++;
    }
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

    comp_stats stats;

    for (uint32_t batch = 0; batch < BATCH_COUNT; batch++)
    {
        print_progress_bar((float)batch / BATCH_COUNT);
        initialize_batch(batch, x);

        map<ptx_instruction::RCP_APPROX_F32><<<GRID_DIM, BLOCK_DIM>>>(BATCH_SIZE, x);
        CUDA_CHECK(cudaPeekAtLastError());

        CUDA_CHECK(cudaDeviceSynchronize());

        compare_batch(batch, x, model_rcp, stats);
    }

    print_progress_bar(1.0f);
    putchar('\n');

    printf("Finite error: max=%.15f, avg=%.15f\n", stats.error.max, stats.error.average());
    printf("Bit-exact: %llu\n", stats.num_exact);

    CUDA_CHECK(cudaFree(x));

    return 0;
}
