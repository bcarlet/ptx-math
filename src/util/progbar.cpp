#include "progbar.hpp"

progbar::progbar(std::string prefix, int width) :
    fraction(0.0f),
    prefix(prefix),
    width(width)
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
        const char c = (i < bar_chars) ? '#' : ' ';
        stream << c;
    }

    stream << '[' << percentage << "%]" << std::flush;

    return stream;
}
