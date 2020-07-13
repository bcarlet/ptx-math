#include "testing.hpp"
#include "batching.hpp"

void test(const interval &sub, float *ref_buf, float *model_buf, uint32_t buf_size,
          const mapf_t &ref, const mapf_t &model, const syncf_t &sync, counters &results)
{
    float start = sub.least;

    while (true)
    {
        uint32_t size = initialize_batch(sub, start, start, buf_size, ref_buf, model_buf);

        if (size == 0u)
            break;

        ref(size, ref_buf);
        model(size, model_buf);

        sync();

        results.accumulate(size, ref_buf, model_buf);
    }
}
