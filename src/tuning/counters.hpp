#ifndef TUNING_COUNTERS_HPP
#define TUNING_COUNTERS_HPP

#include <cstdint>

class counters
{
public:
    enum sign
    {
        NEGATIVE,
        POSITIVE,
        UNDEFINED
    };

    void accumulate(float reference, float model);

    sign first() const;
    sign last() const;

    uint64_t exact() const;
    uint64_t total() const;
    uint64_t regions() const;

private:
    sign m_last = UNDEFINED;
    uint64_t m_exact = 0;
    uint64_t m_total = 0;
    uint64_t m_regions = 0;
};

#endif
