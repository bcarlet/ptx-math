#include "testing.hpp"
#include "batching.hpp"

void test(const interval &test_space, float *gpubuf, float *modelbuf, uint32_t bufsize,
          const mapf_t &gpu, const syncf_t &gpusync, const mapf_t &model, counters &results)
{
    float start = test_space.least;

    while (true)
    {
        uint32_t size = initialize_batch(test_space, start, start, bufsize, gpubuf, modelbuf);

        if (size == 0u)
            break;

        gpu(size, gpubuf);
        model(size, modelbuf);

        gpusync();

        results.accumulate(size, gpubuf, modelbuf);
    }
}
