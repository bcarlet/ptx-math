#include "testing.hpp"
#include "batching.hpp"

void test(const interval &sub, float *reference_buf, float *model_buf, std::size_t buf_size,
          const map_fn &reference, const map_fn &model, const sync_fn &sync, counters &results)
{
    float start = sub.least;

    while (true)
    {
        std::size_t size;
        std::tie(size, start) = initialize_batch(start, sub.greatest, reference_buf, model_buf, buf_size);

        if (size == 0u)
            break;

        reference(size, reference_buf);
        model(size, model_buf);

        sync();

        results.accumulate(size, reference_buf, model_buf);
    }
}
