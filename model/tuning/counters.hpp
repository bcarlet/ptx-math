#ifndef COUNTERS_HPP
#define COUNTERS_HPP

#include <cstddef>
#include <cstdint>

struct counters
{
    void accumulate(std::size_t n, const float *lhs, const float *rhs);

    uint64_t smaller, larger, exact, total;
};

#endif
