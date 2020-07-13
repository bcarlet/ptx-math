#ifndef BASIC_COUNTERS_HPP
#define BASIC_COUNTERS_HPP

#include <cstdint>

#include "base/counters.hpp"

class basic_counters final : public counters
{
public:
    enum err_sign
    {
        NEGATIVE, POSITIVE, UNDEFINED
    };

    void accumulate(std::size_t n, const float *ref, const float *model) override;

    uint64_t exact = 0;
    uint64_t total = 0;
    uint64_t regions = 0;
    err_sign last_sign = UNDEFINED;
};

void clear(basic_counters &count);

#endif
