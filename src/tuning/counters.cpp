#include "counters.hpp"

void counters::accumulate(float reference, float model)
{
    if (reference == model)
    {
        m_exact++;
    }
    else
    {
        const sign cmp = (model < reference) ? NEGATIVE : POSITIVE;

        if (m_last != cmp)
        {
            m_regions++;
            m_last = cmp;
        }
    }

    m_total++;
}

counters::sign counters::first() const
{
    if (m_last == UNDEFINED)
        return UNDEFINED;

    if (m_regions % 2u == 0u)
        return (m_last == NEGATIVE) ? POSITIVE : NEGATIVE;
    else
        return m_last;
}

counters::sign counters::last() const
{
    return m_last;
}

uint64_t counters::exact() const
{
    return m_exact;
}

uint64_t counters::total() const
{
    return m_total;
}

uint64_t counters::regions() const
{
    return m_regions;
}
