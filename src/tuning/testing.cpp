#include "testing.hpp"
#include "batching.hpp"

tester::tester(map_t reference, sync_t sync, float *reference_buf,
               float *model_buf, std::size_t buf_size) :
    reference(reference),
    sync(sync),
    reference_buf(reference_buf),
    model_buf(model_buf),
    buf_size(buf_size)
{
}

void tester::test(float first, float last, const map_t &model, counter &results) const
{
    batcher batch(first, last, reference_buf, model_buf, buf_size);

    while (true)
    {
        const std::size_t size = batch.init_next();

        if (size == 0)
            break;

        reference(size, reference_buf);
        model(size, model_buf);

        sync();

        results.accumulate(size, reference_buf, model_buf);
    }
}
