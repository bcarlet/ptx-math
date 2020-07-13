#ifndef FNTYPES_HPP
#define FNTYPES_HPP

#include <cstddef>
#include <functional>

using mapf_t = std::function<void(std::size_t, float *)>;
using syncf_t = std::function<void()>;

template<class... Args>
using genf_t = std::function<mapf_t(Args...)>;

#endif
