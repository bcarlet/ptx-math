#ifndef UTIL_PROGBAR_HPP
#define UTIL_PROGBAR_HPP

#include <ostream>
#include <string>

class progbar
{
public:
    /**
     * Construct a progress bar with zero progress.
     */
    explicit progbar(std::string prefix = "Progress: ", int width = 50);

    /**
     * Set the progress to a fraction in the range [0, 1].
     */
    progbar &update(float fract);

    /**
     * Insert the progress bar into the given stream, overwriting the current
     * line.
     */
    friend std::ostream &operator<<(std::ostream &stream, const progbar &bar);

private:
    std::string prefix;
    int width;
    float fraction;
};

#endif
