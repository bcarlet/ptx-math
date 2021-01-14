#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>
#include <utility>
#include <string>
#include <map>

using float_limits = std::numeric_limits<float>;

// sanity checks
static_assert(float_limits::is_iec559, "float not IEEE 754");
static_assert(!float_limits::traps, "floating-point exceptions enabled");

#include "ptx/ptx.cuh"
#include "ptxm/models.h"
#include "util/cuda.cuh"
#include "util/progbar.hpp"
#include "util/pun.hpp"

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
    const uint32_t base_val = batch * BATCH_SIZE;

    for (uint32_t i = 0; i < BATCH_SIZE; i++)
    {
        x[i] = pun<float>(base_val + i);
    }
}

static uint32_t compare_batch(uint32_t batch, const float *x, float (*f)(float))
{
    const uint32_t base_val = batch * BATCH_SIZE;
    uint32_t matching = 0;

    for (uint32_t i = 0; i < BATCH_SIZE; i++)
    {
        const float val = pun<float>(base_val + i);
        const float cmp = f(val);

        if (!std::memcmp(x + i, &cmp, sizeof(float)))
            matching++;
    }

    return matching;
}

static uint64_t validate(void (*f)(int, float *), float (*g)(float),
                         bool display_progress = true)
{
    uint64_t matching = 0;
    progbar progress;

    float *x;
    CUDA_CHECK(cudaMallocManaged(&x, BATCH_SIZE * sizeof(float)));

    for (uint32_t batch = 0; batch < BATCH_COUNT; batch++)
    {
        if (display_progress)
            std::cout << progress.update((float)batch / BATCH_COUNT);

        initialize_batch(batch, x);

        f<<<GRID_DIM, BLOCK_DIM>>>(BATCH_SIZE, x);
        CUDA_CHECK(cudaPeekAtLastError());

        CUDA_CHECK(cudaDeviceSynchronize());

        matching += compare_batch(batch, x, g);
    }

    if (display_progress)
        std::cout << progress.update(1.0f) << '\n';

    CUDA_CHECK(cudaFree(x));

    return matching;
}

using model_pair = std::pair<void (*)(int, float *), float (*)(float)>;
using model_map = std::map<std::string, model_pair>;

static std::ostream &operator<<(std::ostream &stream, const cudaDeviceProp &prop)
{
    return stream << prop.name << " (sm_" << prop.major << prop.minor << ')';
}

static void usage [[noreturn]] (const char *prog_name, const model_map &functions)
{
    std::cerr << "Usage: " << prog_name << " [-d <device-number>] <function>\n";
    std::cerr << "\nAvailable functions:\n";

    for (const auto &node : functions)
    {
        std::cerr << "  " << node.first << '\n';
    }

    std::cerr << "\nAvailable devices:\n";

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, i));

        std::cerr << "  Device " << i << ": " << props << '\n';
    }

    std::exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    model_map functions;

    functions["rcp_sm5x"] = std::make_pair(map<ptx_instruction::RCP_APPROX_F32>, ptxm_rcp_sm5x);
    functions["sqrt_sm5x"] = std::make_pair(map<ptx_instruction::SQRT_APPROX_F32>, ptxm_sqrt_sm5x);
    functions["sqrt_sm6x"] = std::make_pair(map<ptx_instruction::SQRT_APPROX_F32>, ptxm_sqrt_sm6x);
    functions["rsqrt_sm5x"] = std::make_pair(map<ptx_instruction::RSQRT_APPROX_F32>, ptxm_rsqrt_sm5x);
    functions["sin_sm5x"] = std::make_pair(map<ptx_instruction::SIN_APPROX_F32>, ptxm_sin_sm5x);
    functions["sin_sm70"] = std::make_pair(map<ptx_instruction::SIN_APPROX_F32>, ptxm_sin_sm70);
    functions["cos_sm5x"] = std::make_pair(map<ptx_instruction::COS_APPROX_F32>, ptxm_cos_sm5x);
    functions["cos_sm70"] = std::make_pair(map<ptx_instruction::COS_APPROX_F32>, ptxm_cos_sm70);
    functions["lg2_sm5x"] = std::make_pair(map<ptx_instruction::LG2_APPROX_F32>, ptxm_lg2_sm5x);
    functions["ex2_sm5x"] = std::make_pair(map<ptx_instruction::EX2_APPROX_F32>, ptxm_ex2_sm5x);

    const char *device_opt = nullptr;
    const char *function_opt = nullptr;

    for (int i = 1; i < argc; i++)
    {
        if (!std::strcmp(argv[i], "-d") && !device_opt)
        {
            if (i + 1 < argc)
                device_opt = argv[++i];
            else
                usage(argv[0], functions);
        }
        else if (!function_opt)
        {
            function_opt = argv[i];
        }
        else
        {
            usage(argv[0], functions);
        }
    }

    if (device_opt)
    {
        int device;

        try {
            device = std::stoi(device_opt);
        } catch (...) {
            usage(argv[0], functions);
        }

        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));

        if (device < 0 || device >= device_count)
            usage(argv[0], functions);

        CUDA_CHECK(cudaSetDevice(device));
    }

    const model_pair *model;

    if (function_opt)
    {
        const auto lookup = functions.find(function_opt);

        if (lookup != functions.end())
            model = &lookup->second;
        else
            usage(argv[0], functions);
    }
    else
    {
        usage(argv[0], functions);
    }

    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::cout << "Using device: " << props << '\n';
    std::cout << "Testing function: " << function_opt << '\n';
    std::cout << "Running simulation...\n";

    const uint64_t matching = validate(model->first, model->second);

    std::cout << "Matching: " << matching << '\n';

    if (matching == (UINT64_C(1) << 32))
        std::cout << "Result: OK\n";
    else
        std::cout << "Result: FAIL\n";

    return EXIT_SUCCESS;
}
