#ifndef TUNING_TESTING_HPP
#define TUNING_TESTING_HPP

#include <functional>

#include "counter.hpp"

class tester
{
public:
    using map_t = std::function<void (std::size_t, float *)>;
    using sync_t = std::function<void ()>;

    tester(map_t reference, sync_t sync, float *reference_buf,
           float *model_buf, std::size_t buf_size);

    void test(float first, float last, const map_t &model, counter &results) const;

private:
    map_t reference;
    sync_t sync;
    float *reference_buf, *model_buf;
    std::size_t buf_size;
};

#endif
