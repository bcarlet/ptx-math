#include "test_handler.hpp"
#include "base/testing.hpp"

test_fn make_test_handler(const interval &sub, const map_fn &reference, const sync_fn &sync,
                          float *reference_buf, float *model_buf, std::size_t buf_size)
{
    return [=](const map_fn &model) -> basic_counters
    {
        basic_counters results;

        test(sub, reference_buf, model_buf, buf_size, reference, model, sync, results);

        return results;
    };
}
