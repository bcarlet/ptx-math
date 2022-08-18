#ifndef TUNING_COMMON_HPP
#define TUNING_COMMON_HPP

#include <cstdint>
#include <functional>

namespace tuning
{

using bias_t        = std::uint64_t;
using coefficient_t = std::uint32_t;
using count_t       = std::uint64_t;

struct error
{
    enum fuzzy_sign: bool
    {
        NON_NEGATIVE,
        NON_POSITIVE
    };

    count_t     regions;    // 0 indicates no observed error
    fuzzy_sign  rightmost;
};

template<class... Args>
using eval_t = std::function<error (Args...)>;

}   // namespace tuning

#endif
