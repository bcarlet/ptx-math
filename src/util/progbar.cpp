#include "progbar.hpp"

#include <ostream>
#include <utility>

namespace util
{

progbar::progbar(std::string prefix, int width) :
    prefix(std::move(prefix)),
    width(width),
    fraction(0.0f)
{
}

progbar &progbar::update(float fract)
{
    fraction = fract;
    return *this;
}

std::ostream &operator<<(std::ostream &stream, const progbar &bar)
{
    const int bar_chars = static_cast<int>(bar.fraction * bar.width);
    const int percentage = static_cast<int>(bar.fraction * 100.0f);

    stream << '\r' << bar.prefix;

    for (int i = 0; i < bar.width; i++)
    {
        stream << ((i < bar_chars) ? '#' : ' ');
    }

    return stream << '[' << percentage << "%]" << std::flush;
}

}   // namespace util
