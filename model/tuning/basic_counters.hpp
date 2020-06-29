#ifndef BASIC_COUNTERS_HPP
#define BASIC_COUNTERS_HPP

#include "base/counters.hpp"

#include <cstdint>

class basic_counters final : public counters
{
public:
    void accumulate(std::size_t n, const float *gpu, const float *model) override;

    void clear();

    uint64_t smaller, larger, exact, total;
};

#endif
