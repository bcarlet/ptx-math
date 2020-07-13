#ifndef FNTYPES_HPP
#define FNTYPES_HPP

#include <cstddef>
#include <functional>

using map_fn = std::function<void(std::size_t, float *)>;
using sync_fn = std::function<void()>;

template<class... Args>
using gen_fn = std::function<map_fn(Args...)>;

#endif
