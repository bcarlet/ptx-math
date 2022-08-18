#ifndef UTIL_STOPWATCH_HPP
#define UTIL_STOPWATCH_HPP

#include <chrono>

namespace util
{

class stopwatch
{
public:
    using clock = std::chrono::steady_clock;

    /**
     * Construct and start a stopwatch.
     */
    stopwatch() :
        begin(clock::now())
    {
    }

    /**
     * Reset the elapsed time.
     */
    void reset()
    {
        begin = clock::now();
    }

    /**
     * Get the time elapsed since the stopwatch's creation or last reset.
     */
    template<class Rep = double, class Period = std::ratio<1>>
    Rep elapsed() const
    {
        using duration = std::chrono::duration<Rep, Period>;

        const clock::time_point end = clock::now();
        const duration diff = std::chrono::duration_cast<duration>(end - begin);

        return diff.count();
    }

private:
    clock::time_point begin;
};

}   // namespace util

#endif
