#ifndef COUNTERS_HPP
#define COUNTERS_HPP

#include <cstddef>

class counters
{
public:
    virtual void accumulate(std::size_t n, const float *ref, const float *model) = 0;

protected:
    ~counters() = default;
};

#endif
