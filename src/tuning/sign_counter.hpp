#ifndef TUNING_SIGN_COUNTER_HPP
#define TUNING_SIGN_COUNTER_HPP

#include <cstdint>

#include "counter.hpp"

class sign_counter final : public counter
{
public:
    enum error_sign
    {
        NEGATIVE, POSITIVE, UNDEFINED
    };

    void accumulate(std::size_t n, const float *reference, const float *model) override;

    void clear();

    error_sign first() const;
    error_sign last() const;

    uint64_t exact = 0;
    uint64_t total = 0;
    uint64_t regions = 0;

private:
    error_sign last_sign = UNDEFINED;
};

#endif
