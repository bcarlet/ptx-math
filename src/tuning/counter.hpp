#ifndef TUNING_COUNTER_HPP
#define TUNING_COUNTER_HPP

#include <cstddef>

class counter
{
public:
    virtual void accumulate(std::size_t n, const float *reference, const float *model) = 0;

protected:
    ~counter() = default;
};

#endif
