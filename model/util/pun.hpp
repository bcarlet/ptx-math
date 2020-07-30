#ifndef UTIL_PUN_HPP
#define UTIL_PUN_HPP

#include <cstring>

/**
 * Reinterpret the bit representation of a value of type From as a value of type
 * To.
 */
template<class To, class From>
inline To pun(const From &x)
{
    static_assert(sizeof(To) == sizeof(From), "sizeof(To) must equal sizeof(From)");

    To r;
    std::memcpy(&r, &x, sizeof(To));

    return r;
}

#endif
