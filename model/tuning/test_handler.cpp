#include "test_handler.hpp"
#include "base/testing.hpp"

testf_t make_test_handler(const interval &sub, const mapf_t &ref, const syncf_t &sync,
                          float *ref_buf, float *model_buf, std::size_t buf_size)
{
    return [=](const mapf_t &model) -> basic_counters
    {
        basic_counters results;

        test(sub, ref_buf, model_buf, buf_size, ref, model, sync, results);

        return results;
    };
}
